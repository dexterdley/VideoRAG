import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration

# Qwen2.5-VL uses its own processor; no manual normalisation needed.
# qwen_vl_utils is required for video inputs but optional for single images.
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Zero-Shot Yes/No Inference")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Local path or HuggingFace ID (e.g. Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str,
                        default="Is this a key highlight? Answer only Yes or No",
                        help="The Yes/No question to ask")
    parser.add_argument("--max_new_tokens", type=int, default=15,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2. Load Model
    # AutoModelForImageTextToText covers all Qwen2.5-VL sizes natively in transformers ≥4.52
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # 3. Extract Token IDs Dynamically
    temp_ids = processor.tokenizer(["Yes", "No"], add_special_tokens=False).input_ids
    yes_id = temp_ids[0][0]
    no_id  = temp_ids[1][0]

    # 4. Build Conversation-Style Messages (Qwen2.5-VL native format)
    image = Image.open(args.image_path).convert("RGB")
    msgs = [
        {
            "role": "system",
            "content": "You are an expert video editor. Strictly answer only Yes or No.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": args.question},
            ],
        },
    ]

    # 5. Apply Chat Template and Pre-process
    text = processor.apply_chat_template(
        msgs, tokenize=False, enable_thinking=False, add_generation_prompt=True
    )

    if HAS_QWEN_UTILS:
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
    else:
        # Fallback: pass the PIL image directly (works for single images)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)

    with torch.inference_mode():
        # 6a. Generate text response
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        # 6b. Compute logits at the last prompt token for zero-shot scoring
        outputs = model(**inputs)

    # 7a. Decode generated text (strip prompt tokens)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_length:]
    generated_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # 7b. Process logits
    next_token_logits = outputs.logits[0, -1, :]
    yes_logit = next_token_logits[yes_id]
    no_logit  = next_token_logits[no_id]

    # Relative normalised probability between Yes and No
    confidences = F.softmax(torch.tensor([yes_logit, no_logit], dtype=torch.float32), dim=0)
    yes_conf = confidences[0].item()
    no_conf  = confidences[1].item()

    # 8. Output Results
    print("-" * 40)
    print(f"Image Size          : {image.size} (W x H)")
    print(f"Token Index [Yes]   : {yes_id}")
    print(f"Token Index [No]    : {no_id}")
    print("-" * 40)
    print(f"Question     : {args.question}")
    print(f"Model Output : {generated_text}")
    print("-" * 40)
    print(f"Yes Conf     : {yes_conf:.2%}")
    print(f"No Conf      : {no_conf:.2%}")
    print(f"Verdict      : {'Yes' if yes_conf > no_conf else 'No'}")


if __name__ == "__main__":
    main()
