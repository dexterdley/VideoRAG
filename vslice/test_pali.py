import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description="PaliGemma2 Zero-Shot Yes/No Inference")
    parser.add_argument("--model_path", type=str, default="/home/dexter/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots/96eeb174da13ca1a2b247e4d0867436296c36420/", help="Local path or HuggingFace ID")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input .jpg")
    parser.add_argument("--question", type=str, default="Is this a key highlight? Answer only Yes or No", help="The Yes/No question to ask")
    parser.add_argument("--max_new_tokens", type=int, default=15, help="Number of tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Processor and Model
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    model.eval()

    # 2. Extract Token IDs Dynamically
    # add_special_tokens=False prevents injecting <bos> tokens into the lookup
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

    # 3. Format Prompt
    # <image> must be at the very start, \n at the very end to trigger generation
    prompt = f"<image>answer en {args.question}\n"
    image = Image.open(args.image_path).convert("RGB")

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        # 5a. Generate the Text Response
        # We use greedy decoding (do_sample=False) for deterministic QA
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens, 
            do_sample=False
        )
        
        # 5b. Extract the Logits for Zero-Shot scoring
        outputs = model(**inputs)
        
    # 6a. Decode generated text
    input_length = inputs["input_ids"].shape[1]
    # Slice off the prompt tokens so we only decode the new response
    new_tokens = generated_ids[0][input_length:]
    generated_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
        
    # 6b. Process Logits
    next_token_logits = outputs.logits[0, -1, :]
    yes_logit = next_token_logits[yes_id]
    no_logit = next_token_logits[no_id]
    
    # Relative normalized probability between Yes and No
    confidences = F.softmax(torch.tensor([yes_logit, no_logit], dtype=torch.float32), dim=0)
    yes_conf = confidences[0].item()
    no_conf = confidences[1].item()

    # 7. Output Results
    print("-" * 40)
    print(f"Reshaped Image Size : {image.size} (W x H)")
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