import torch
import soundfile as sf
import warnings

# 1. Suppress the internal librosa FutureWarnings to keep your console clean
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig

# Manually add the missing default if it's absent
if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, 'use_sliding_window'):
    Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = False

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "./Qwen3-Omni-30B-A3B-Instruct"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# 2. FIX: Monkey-patch the talker to ensure it auto-moves incoming tensors to its own GPU (e.g., cuda:6)
if hasattr(model, "talker"):
    original_talker_generate = model.talker.generate
    def patched_talker_generate(*args, **kwargs):
        talker_device = next(model.talker.parameters()).device
        # Move all tensor kwargs/args to the talker's specific device
        patched_kwargs = {k: v.to(talker_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        patched_args = tuple(a.to(talker_device) if isinstance(a, torch.Tensor) else a for a in args)
        return original_talker_generate(*patched_args, **patched_kwargs)
    
    model.talker.generate = patched_talker_generate

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
            {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
            {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
        ],
    },
]

USE_AUDIO_IN_VIDEO = True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)

# Safely move inputs to device. (Avoid .to(model.dtype) on the whole dict, 
# as it can mistakenly convert integer masks/IDs into floats)
inputs = inputs.to(model.device)
for k, v in inputs.items():
    if torch.is_tensor(v) and torch.is_floating_point(v):
        inputs[k] = v.to(model.dtype)

outputs = model.generate(
    **inputs, 
    # speaker="Ethan",  <-- REMOVED: This disables the talker/TTS module
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id
)

# When TTS is disabled, generate() usually just returns the text tensor directly, 
# rather than a (text_ids, audio) tuple. Let's handle it safely:
text_ids = outputs[0] if isinstance(outputs, tuple) else outputs

# Decode the text
text = processor.batch_decode(
    text_ids[:, inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(text)

'''
# Inference
text_ids, audio = model.generate(
    **inputs, 
    #speaker="Ethan", 
    thinker_return_dict_in_generate=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id
    )

text = processor.batch_decode(text_ids[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
print(text)

if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
'''