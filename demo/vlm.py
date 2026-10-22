"""
Thin wrapper around the local MiniCPM-V 2.6 (int4) base model.

Three primitives are all the demo needs:
  * score_frames : batched single-image Yes/No probing -> P(Yes) per frame
  * score_clip   : multi-image Yes/No probing         -> P(Yes) for a clip
  * chat         : free-form generation (captions, music mood)

Optional LoRA adapters (PEFT folders, e.g. the DPO checkpoints written by
vslice/simple_dpo.py) can be attached on top of the resident int4 model and
swapped at runtime with `set_lora(path)` without reloading the base weights.

An adapter drives the per-frame scoring (what the DPO adapters were trained on);
clip checks and captions / title / mood stay on the base model unless
`lora_all=True`. DPO only learns *relative* preferences, so an adapter's absolute
P(Yes) can drift far from the base model's (the SumMe DPO splits put almost every
frame at P(Yes) < 1e-3). `calibrate()` maps the adapter's Yes-No log-odds back onto
the base model's scale with an affine fit on a few anchor frames; the frame
ordering stays the adapter's own.
"""

import contextlib
import os

import numpy as np
import torch
from tqdm import tqdm

from samples import REPO_ROOT, describe_lora, is_lora_dir

DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "MiniCPM-V-2_6-int4")
IMAGE_TAG = "(<image>./</image>)"
YES_NO_SYSTEM = "You are an expert video editor. Strictly answer only Yes or No."


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class MiniCPM:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="cuda", lora_path=None, lora_all=False):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPM-V int4 (bitsandbytes) needs a CUDA GPU.")
        print(f"[MiniCPM] loading {model_path} ...")
        # The checkpoint already carries a bnb nf4 quantization_config -> no dtype / bnb args needed.
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device_map=device, attn_implementation="sdpa"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
        self.yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"[MiniCPM] ready ({torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU)")

        self.peft = None        # PeftModel wrapper, created on the first adapter load
        self.adapters = {}      # realpath -> adapter name (loaded adapters stay cached on the GPU)
        self.lora_path = None   # active adapter, None = base model
        self.lora_all = lora_all
        self.calib = None       # (a, b): frame log-odds -> a * logodds + b, for the active adapter
        if lora_path:
            self.set_lora(lora_path)

    # ───────────────────────── LoRA adapters ─────────────────────────

    def set_lora(self, path=None, lora_all=None):
        """
        Activate the PEFT adapter at `path` (loaded once, then cached), or go back
        to the plain base model when `path` is empty. Returns a short description.
        """
        if lora_all is not None:
            self.lora_all = bool(lora_all)
        self.calib = None
        if not path:
            if self.peft is not None and self.lora_path:
                self.peft.base_model.disable_adapter_layers()
            self.lora_path = None
            return "base model (no LoRA)"

        if not is_lora_dir(path):
            raise FileNotFoundError(f"Not a PEFT/LoRA adapter folder (no adapter_config.json): {path}")
        key = os.path.realpath(path)
        if key not in self.adapters:
            from peft import PeftModel

            name = f"lora{len(self.adapters)}"
            print(f"[MiniCPM] loading LoRA {path} ...")
            # LoRA layers are injected into self.model in place, so the forward / chat
            # calls below pick them up without going through the wrapper.
            if self.peft is None:
                self.peft = PeftModel.from_pretrained(self.model, path, adapter_name=name, is_trainable=False)
            else:
                self.peft.load_adapter(path, adapter_name=name, is_trainable=False)
            self.peft.eval()
            self.adapters[key] = name
        self.peft.base_model.enable_adapter_layers()
        self.peft.set_adapter(self.adapters[key])
        self.lora_path = path
        return f"{os.path.basename(os.path.normpath(path))} ({describe_lora(path)})"

    def _base_only(self):
        """Context for calls that should bypass the adapter (unless lora_all)."""
        if self.lora_path and not self.lora_all:
            return self.peft.disable_adapter()
        return contextlib.nullcontext()

    def calibrate(self, frames, question, n=12, batch_size=4, system_prompt=YES_NO_SYSTEM):
        """
        Fit base_logodds ~= a * adapter_logodds + b on `n` evenly spaced frames and use it
        in score_frames until the next set_lora / calibrate. n=0 (or no adapter) clears it.
        Returns (a, b) or None.
        """
        self.calib = None
        if not self.lora_path or n <= 0 or len(frames) < 3:
            return None
        idx = np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int))
        anchors = [[frames[i]] for i in idx]
        lora = self._log_odds(anchors, question, system_prompt, batch_size)
        with self.peft.disable_adapter():
            base = self._log_odds(anchors, question, system_prompt, batch_size)
        a, b = np.polyfit(lora, base, 1) if np.ptp(lora) > 1e-6 else (0.0, 0.0)
        if not (np.isfinite(a) and np.isfinite(b) and a > 0):
            a, b = 1.0, float(np.mean(base - lora))  # degenerate fit -> shift only
        self.calib = (float(a), float(b))
        return self.calib

    # ───────────────────────── yes / no probing ─────────────────────────

    def _log_odds(self, image_lists, question, system_prompt, batch_size=None):
        """logit(Yes) - logit(No) at the answer position; sigmoid of it == softmax P(Yes)."""
        batch_size = batch_size or len(image_lists)
        out = []
        for start in range(0, len(image_lists), batch_size):
            prompts, imgs = [], image_lists[start:start + batch_size]
            for frames in imgs:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": IMAGE_TAG * len(frames) + "\n" + question},
                ]
                prompts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            inputs = self.processor(
                prompts, imgs, max_slice_nums=1, use_image_id=False,
                return_tensors="pt", max_length=4096,
            ).to(self.device)
            inputs.pop("image_sizes", None)
            mask = inputs["attention_mask"].long()
            # Processor left-pads -> positions must start at 0 on the first real token.
            inputs["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)

            with torch.inference_mode():
                logits = self.model(inputs, attention_mask=mask).logits[:, -1, :].float()
            out.append((logits[:, self.yes_id] - logits[:, self.no_id]).cpu().numpy())
        return np.concatenate(out)

    def score_frames(self, frames, question, batch_size=4, system_prompt=YES_NO_SYSTEM, progress=None):
        """P(Yes) for every frame (single-image prompts, batched; calibrated if an adapter is active)."""
        scores = []
        steps = range(0, len(frames), batch_size)
        for n, start in enumerate(tqdm(steps, desc="Scoring frames", disable=len(steps) <= 2)):
            z = self._log_odds([[f] for f in frames[start:start + batch_size]], question, system_prompt)
            if self.calib and self.lora_path:
                z = self.calib[0] * z + self.calib[1]
            scores.extend(sigmoid(z))
            if progress is not None:
                progress((n + 1) / len(steps))
        return scores

    def score_clip(self, frames, question, system_prompt=YES_NO_SYSTEM):
        """P(Yes) for a clip given a few of its frames in one prompt."""
        with self._base_only():
            return float(sigmoid(self._log_odds([list(frames)], question, system_prompt))[0])

    # ───────────────────────── generation ─────────────────────────

    def chat(self, frames, text, max_new_tokens=48, system_prompt=None):
        msgs = [{"role": "user", "content": list(frames) + [text]}]
        with self._base_only():
            out = self.model.chat(
                image=None, msgs=msgs, tokenizer=self.tokenizer, sampling=False,
                max_new_tokens=max_new_tokens, max_slice_nums=1, use_image_id=False,
                system_prompt=system_prompt or "",
            )
        return out.strip()
