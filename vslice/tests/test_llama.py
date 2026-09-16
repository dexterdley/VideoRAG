import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Llama-3.2-Vision uses MllamaForConditionalGeneration natively in transformers ≥4.45.
# No trust_remote_code, no custom chat helpers needed.


def main():
    parser = argparse.ArgumentParser(description="Llama-3.2-Vision Zero-Shot Yes/No Inference")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        help="Local path or HuggingFace ID (e.g. meta-llama/Llama-3.2-11B-Vision-Instruct)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str,
                        default="Is this a key highlight? Answer only Yes or No",
                        help="The Yes/No question to ask")
    parser.add_argument("--max_new_tokens", type=int, default=15,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Processor
    # MllamaProcessor wraps the Llama tokenizer + image transforms in one object.
    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2. Load Model
    # MllamaForConditionalGeneration is the native class; AutoModelForImageTextToText
    # also resolves to it, either works.
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # 3. Extract Token IDs Dynamically
    # Llama-3 tokeniser: "Yes" → single token; "No" → single token.
    temp_ids = processor.tokenizer(["Yes", "No"], add_special_tokens=False).input_ids
    yes_id = temp_ids[0][0]
    no_id  = temp_ids[1][0]

    # 4. Build Conversation-Style Messages (Llama-3.2-Vision native format)
    # The <|image|> special token must appear in the user turn for cross-attention to fire.
    image = Image.open(args.image_path).convert("RGB")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},          # placeholder; image supplied separately
                {"type": "text", "text": args.question},
            ],
        }
    ]

    # 5. Apply Chat Template and Pre-process
    # apply_chat_template inserts the <|image|> token at the correct position.
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)

    inputs = processor(
        text=text,
        images=[image],       # list: one image per <|image|> placeholder
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
