import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    GenerationMixin,
    GenerationConfig,
    BitsAndBytesConfig,
)
import transformers.cache_utils as cache_utils
from qwen_vl_utils import process_vision_info
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────── MODEL LOADING ────────────────────────

class QwenVLWrapper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

def load_vlm(model_path, model_type, device, load_in_4bit=False):
    """Load specified VLM and return (model, tokenizer, processor, yes_id, no_id)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[{device}] Loading {model_type} from {model_path}...")

    # Configure 4-bit quantization config if requested
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    if model_type == "minicpm":
        kwargs = {
            "trust_remote_code": True,
            "device_map": device,
            "attn_implementation": "sdpa" if load_in_4bit else "eager",
        }
        if load_in_4bit:
            kwargs["quantization_config"] = bnb_config
            # When using bitsandbytes, device_map should be "auto" or device index
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = dtype
            
        model = AutoModel.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            model = model.eval()
            
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]
        print(f"[{device}] [OK] MiniCPM Loaded (Yes={yes_id}, No={no_id})")
        return model, tokenizer, processor, yes_id, no_id

    elif model_type == "qwen3":
        kwargs = {
            "trust_remote_code": True,
            "device_map": device,
            "attn_implementation": "sdpa" if load_in_4bit else "eager",
        }

        if load_in_4bit:
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = dtype
            
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            model = model.eval()
            
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        # In Qwen, 'Yes' and 'No' IDs
        temp_ids = processor.tokenizer(["Yes", "No"], add_special_tokens=False).input_ids
        yes_id = temp_ids[0][0]
        no_id = temp_ids[1][0]
        
        print(f"[{device}] [OK] Qwen VL Loaded (Yes={yes_id}, No={no_id})")
        return model, processor.tokenizer, processor, yes_id, no_id

    elif "qwen2" in model_type:
        kwargs = {
            "trust_remote_code": True,
            "device_map": device,
            "attn_implementation": "sdpa",
        }

        if load_in_4bit:
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = dtype
            
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            model = model.eval()
            
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
            trust_remote_code=True
        )
        temp_ids = processor.tokenizer(["Yes", "No"], add_special_tokens=False).input_ids
        yes_id = temp_ids[0][0]
        no_id = temp_ids[1][0]
        
        print(f"[{device}] [OK] Qwen2.5-VL Loaded (Yes={yes_id}, No={no_id})")
        return model, processor.tokenizer, processor, yes_id, no_id

    elif model_type == "moondream":
        kwargs = {
            "trust_remote_code": True,
            "device_map": device,
        }
        revision = "2024-08-26" if model_path == "vikhyatk/moondream2" else None
        if revision:
            kwargs["revision"] = revision

        if load_in_4bit:
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = dtype

        # Compatibility shims for Moondream in transformers >= 4.50
        if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
            cache_utils.DynamicCache.get_usable_length = lambda self, seq_len=None, layer_idx=0: self.get_seq_length(layer_idx)

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            model = model.eval()

        # In transformers >= 4.50, PreTrainedModel no longer inherits from GenerationMixin.
        # Restore generative capabilities to model.text_model if missing.
        if hasattr(model, "text_model"):
            if not hasattr(model.text_model, "generate"):
                cls = model.text_model.__class__
                model.text_model.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
            if getattr(model.text_model, "generation_config", None) is None:
                model.text_model.generation_config = GenerationConfig.from_model_config(model.text_model.config)

        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        processor = tokenizer
        if not hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer
        processor.model_type = "moondream"

        # Attach forward method to Moondream so peft_model(**batch) and model(**batch) work seamlessly
        def moondream_forward(self, input_ids=None, attention_mask=None, pixel_values=None, output_hidden_states=False, **kwargs):
            text_emb = self.text_model.get_input_embeddings()
            B = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]

            if pixel_values is not None:
                # If input_ids begins with <image> tokens ([27, 9060, 29]), strip them so text_embeds starts after <image>
                if input_ids is not None and input_ids.shape[1] >= 3 and (input_ids[:, :3] == torch.tensor([27, 9060, 29], device=input_ids.device)).all():
                    text_tokens = input_ids[:, 3:]
                    text_mask = attention_mask[:, 3:] if attention_mask is not None else None
                else:
                    text_tokens = input_ids
                    text_mask = attention_mask

                bos_id = getattr(self.config, "bos_token_id", None) or 50256
                bos = torch.full((B, 1), bos_id, dtype=torch.long, device=pixel_values.device)
                bos_embeds = text_emb(bos)
                image_embeds = self.vision_encoder(pixel_values)
                text_embeds = text_emb(text_tokens) if text_tokens is not None else None

                embeds = [bos_embeds, image_embeds]
                if text_embeds is not None:
                    embeds.append(text_embeds)
                inputs_embeds = torch.cat(embeds, dim=1)

                if text_mask is not None:
                    img_mask = torch.ones((B, 1 + image_embeds.shape[1]), dtype=text_mask.dtype, device=text_mask.device)
                    full_mask = torch.cat([img_mask, text_mask], dim=1)
                else:
                    full_mask = None

                return self.text_model(inputs_embeds=inputs_embeds, attention_mask=full_mask, output_hidden_states=output_hidden_states, **kwargs)
            else:
                return self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states, **kwargs)

        import types
        model.forward = types.MethodType(moondream_forward, model)

        # Moondream (Phi tokenizer) BPE produces words after "Answer:" with leading space:
        # " Yes" -> 3363, " No" -> 1400
        yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_id  = tokenizer.encode(" No", add_special_tokens=False)[0]

        print(f"[{device}] [OK] Moondream Loaded (Yes={yes_id}, No={no_id})")
        return model, tokenizer, processor, yes_id, no_id

# ─────────────────────── VLM INFERENCE ───────────────────────

def minicpm_extract_title_and_keywords(raw_title, model, processor):
    """Uses MiniCPM to clean the title and expand it into visual keywords once per video."""
    prompt = (
        "Task: Clean the video title and extract 3 visual keywords.\n"
        "Format: <Cleaned Title> | <keyword1>, <keyword2>, <keyword3>\n"
        "Example Input: 'playing basketball (Must Watch!)'\n"
        "Example Output: Playing basketball | basketball, hoop, sports\n\n"
        f"Input: '{raw_title}'\n"
        "Output:"
    )
    
    msgs = [{'role': 'user', 'content': prompt}]
       
    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=processor.tokenizer,
        sampling=False, 
        temperature=0.1
    ).strip()
    
    # Parse the output
    if "|" in res:
        parts = res.split("|")
        # Strip out any accidental prefix the model might add
        cleaned_title = parts[0].replace("Cleaned Title:", "").strip()
        keywords = parts[1].replace("Keywords:", "").strip()
    else:
        # Fallback just in case the model ignores the formatting rules
        print(f"  [WARN] Format failed for '{raw_title}', falling back to raw title.")
        cleaned_title = raw_title
        keywords = res
        
    return cleaned_title, keywords

def qwen_extract_title_and_keywords(raw_title, wrapper):
    """Uses Qwen to clean the title and expand it into visual keywords once per video."""
    prompt = (
        "Task: Clean the video title and extract 3 visual keywords.\n"
        "Format: <Cleaned Title> | <keyword1>, <keyword2>, <keyword3>\n"
        "Example Input: 'playing basketball (Must Watch!)'\n"
        "Example Output: Playing basketball | basketball, hoop, sports\n\n"
        f"Input: '{raw_title}'\n"
        "Output:"
    )
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    # Format the prompt using Qwen's template
    text = wrapper.processor.apply_chat_template(msgs, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    
    # We only pass text here, no images
    inputs = wrapper.processor(text=[text], padding=True, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        # Use .generate() to actually output text tokens
        generated_ids = wrapper.model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False, # Greedy decoding for formatting reliability
            temperature=0.1
        )
        
    # Trim the prompt from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode to string
    res = wrapper.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Parse the output
    if "|" in res:
        parts = res.split("|")
        cleaned_title = parts[0].replace("Cleaned Title:", "").replace("Title:", "").strip()
        keywords = parts[1].replace("Keywords:", "").replace("keywords:", "").strip()
    else:
        print(f"  [WARN] Qwen format failed for '{raw_title}'. Raw output: {res}")
        cleaned_title = raw_title
        keywords = res
        
    return cleaned_title, keywords

def minicpm_inference(images, title, keywords, model, processor, yes_id, no_id, skills=None):
    prompts_lists = []
    input_images_lists = []
    system_prompt = "You are an expert video editor. Strictly answer only Yes or No."
    formatted_prompt = f"Does this image represent the core message of {keywords} in the video context of '{title}'?"
        
    # 1. Build the In-Context Learning (ICL) Prefix
    icl_text = ""
    icl_images = []
    
    if skills:
        icl_text += "### CALIBRATION EXAMPLES\nReview these previous evaluations to avoid common errors:\n"
        for skill in skills:
            error_type = skill['type']
            
            if error_type == 'tp':
                label, explanation = "Yes", "This is a perfect core highlight."
            elif error_type == 'fn':
                label, explanation = "Yes", "This is important action that must be included."
            elif error_type == 'fp':
                label, explanation = "No", "CRITICAL: This is a deceptive frame (background/noise). Output No."
            elif error_type == 'tn':
                label, explanation = "No", "This is correctly identified as a non-highlight."
            
            # Add the text and image placeholder for the example
            icl_text += f"Example Frame (<image>./</image>) from '{skill['title']}': Should this be in the highlight reel? {label}. ({explanation})\n"
            
            # Load the actual image into memory
            icl_images.append(Image.open(skill['image_path']).convert("RGB"))
        
        icl_text += "\nNow, keeping past mistakes in mind, evaluate the following new frame.\n"
    for img in images:
        full_user_text = icl_text + f"(<image>./</image>)\n{formatted_prompt}"
        
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': full_user_text} # Pass the combined text here
        ]
        prompt_str = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts_lists.append(prompt_str)
        
        current_input_images = icl_images + [img] 
        input_images_lists.append(current_input_images)

    inputs = processor(
        prompts_lists,
        input_images_lists,
        max_slice_nums=1,
        use_image_id=False,
        return_tensors="pt",
        max_length=2048
    ).to(device)

    if "position_ids" not in inputs:
        batch_size, seq_len = inputs["input_ids"].shape
        inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

    if "image_sizes" in inputs:
        inputs.pop("image_sizes")

    with torch.inference_mode():
        outputs = model(inputs, attention_mask=inputs.get("attention_mask"), output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states[-2][:, -1, :]

        yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
        binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
        contrast = F.relu(binary_probs[:, 0] - binary_probs[:, 1])
    return binary_probs[:, 0], binary_probs[:, 1], yes_logits, no_logits, hidden_states

def qwen_inference(images, title, keywords, wrapper, yes_id, no_id, skills=None):
    if process_vision_info is None:
        raise ImportError("qwen_vl_utils is required for Qwen inference.")
    
    system_prompt = "You are an expert video editor. Strictly answer only Yes or No."
    formatted_prompt =(f"Does this image represent the core message of the video '{title}'?"
        f"Does this frame showcase key visual elements of ({keywords})? "
    )
    
    probs_yes = []
    probs_no = []
    confs_all = []
    hidden_states_all = []
    
    # 1. Build the In-Context Learning (ICL) Prefix
    icl_text = ""
    icl_images = []
    
    if skills:
        icl_text += "Review these past evaluations to calibrate your judgment:\n"
        for skill in skills:
            # Check the filename to see if this was a TP (Good) or FP (Bad) example
            is_good = "good_" in skill['image_path'].lower()
            
            if is_good:
                label = "Yes"
                explanation = "Yes, this frame provides a good summary."
            else:
                label = "No"
                explanation = "You incorrectly guessed Yes for this in the past. This is actually a boring or irrelevant shot and is No."
            
            # Add the text and image placeholder for the example
            icl_text += f"Example Frame (<image>./</image>) from '{skill['title']}': Should this be in the highlight reel? {label}. ({explanation})\n"
            
            # Load the actual image into memory
            icl_images.append(Image.open(skill['image_path']).convert("RGB"))
        
        icl_text += "\nNow, keeping past mistakes in mind, evaluate the following new frame.\n"

    for img in images:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": formatted_prompt}]}
        ]
        text = wrapper.processor.apply_chat_template(msgs, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = wrapper.processor(text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt",
            max_length=2048
        ).to(device)
        
        with torch.inference_mode():
            outputs = wrapper.model(**inputs, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states[-2][:, -1, :]

            yes_logits, no_logits = logits[:, yes_id], logits[:, no_id]
            binary_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)
            
            probs_yes.append(binary_probs[:, 0])
            probs_no.append(binary_probs[:, 1])
            confs_all.append(F.relu(binary_probs[:, 0] - binary_probs[:, 1]).pow(2))
            hidden_states_all.append(hidden_states)
    
    return torch.cat(probs_yes), torch.cat(probs_no), yes_logits, no_logits, torch.cat(hidden_states_all, dim=0)