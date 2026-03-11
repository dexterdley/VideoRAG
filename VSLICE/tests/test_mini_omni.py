import torch
import requests
import urllib.request
import os
import librosa
from PIL import Image
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# CMD: python ./VSLICE/tests/test_mini_omni.py

model_path = '.checkpoints/MiniCPM-o-2_6-int4'

print("Loading model...")
model = AutoGPTQForCausalLM.from_quantized(
    model_path,
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    trust_remote_code=True,
    disable_exllama=True,
    disable_exllamav2=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
print("Model loaded successfully!")

# --- 1. Fetch and Process Image ---
print("Fetching image...")
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

# --- 2. Fetch and Process Audio ---
print("Fetching audio...")
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
temp_audio_path = "temp_cough.wav"

# Download the file temporarily so librosa can decode it cleanly
urllib.request.urlretrieve(audio_url, temp_audio_path)

# MiniCPM-o strictly requires 16kHz mono audio arrays
audio_data, _ = librosa.load(temp_audio_path, sr=16000, mono=True)
os.remove(temp_audio_path)

# --- 3. Format Omni-Modal Conversation ---
# Pass the raw objects sequentially into the content list
conversation = [
    {
        "role": "user", 
        "content": [
            image, 
            audio_data, 
            "What can you see and hear? Answer in one short sentence."
        ]
    }
]

print("Generating response...")

response = model.chat(
    image=None,
    msgs=conversation,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    omni_input=True  # REQUIRED: Tells the model to expect interleaved audio/visual inputs
)

print("\n--- Model Response ---")
print(response)