import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.cache_utils as cache_utils
import transformers.utils as transformers_utils

# Compatibility shims for environments with transformers < 4.41 or >= 4.50:
if not hasattr(cache_utils, "StaticCache"):
    cache_utils.StaticCache = getattr(cache_utils, "DynamicCache", type("StaticCache", (), {}))
if not hasattr(transformers_utils, "is_torchdynamo_compiling"):
    transformers_utils.is_torchdynamo_compiling = getattr(transformers_utils, "is_torchdynamo_compiling", lambda: False)
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
    cache_utils.DynamicCache.get_usable_length = lambda self, seq_len=None, layer_idx=0: self.get_seq_length(layer_idx)

from transformers import GenerationMixin, GenerationConfig

def main():
    parser = argparse.ArgumentParser(description="Moondream2 Zero-Shot Yes/No Inference")
    parser.add_argument("--model_path", type=str, default="vikhyatk/moondream2",
                        help="Local path or HuggingFace ID (default: vikhyatk/moondream2)")
    parser.add_argument("--revision", type=str, default="2024-08-26",
                        help="Model revision tag on HuggingFace for stability")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str,
                        default="Is this a key highlight? Answer only Yes or No",
                        help="The Yes/No question to ask")
    parser.add_argument("--max_new_tokens", type=int, default=15,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=args.revision if args.revision else None,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=args.revision if args.revision else None,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    # In transformers >= 4.50, PreTrainedModel no longer inherits from GenerationMixin.
    # Restore generative capabilities to model.text_model if missing.
    if hasattr(model, "text_model"):
        if not hasattr(model.text_model, "generate"):
            cls = model.text_model.__class__
            model.text_model.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
        if getattr(model.text_model, "generation_config", None) is None:
            model.text_model.generation_config = GenerationConfig.from_model_config(model.text_model.config)

    # 2. Extract Token IDs Dynamically
    # Moondream (Phi tokenizer) BPE generates words after "Answer:" with a leading space.
    # Token 3363 is " Yes", Token 1400 is " No". (Without space: 5297 is "Yes", 2949 is "No").
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode(" No", add_special_tokens=False)[0]

    # 3. Load and Encode Image
    image = Image.open(args.image_path).convert("RGB")

    with torch.inference_mode():
        # Encode image into visual embeddings (shape: [1, 729, hidden_dim])
        enc_image = model.encode_image(image)

        # 4a. Text Generation
        if hasattr(model, "answer_question"):
            generated_text = model.answer_question(enc_image, args.question, tokenizer=tokenizer)
        elif hasattr(model, "query"):
            res = model.query(image=image, question=args.question)
            generated_text = res.get("answer", str(res))
        else:
            generated_text = "N/A"

        # 4b. Compute logits for zero-shot Yes/No scoring
        # In moondream.py, model.input_embeds() requires the "<image>" tag in prompt to splice in image_embeds!
        # If "<image>" is missing, input_embeds() ignores enc_image entirely and evaluates text-only!
        prompt = f"<image>\n\nQuestion: {args.question}\n\nAnswer:"
        text_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Retrieve text embeddings
        if hasattr(model, "input_embeds"):
            inputs_embeds = model.input_embeds(prompt, enc_image, tokenizer)
            outputs = model.text_model(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[0, -1, :]
        elif hasattr(model, "text_model"):
            text_embeds = model.text_model.get_input_embeddings()(text_tokens)
            inputs_embeds = torch.cat([enc_image, text_embeds], dim=1)
            outputs = model.text_model(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[0, -1, :]
        elif hasattr(model, "transformer"):
            text_embeds = model.transformer.wte(text_tokens)
            inputs_embeds = torch.cat([enc_image, text_embeds], dim=1)
            outputs = model.transformer(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[0, -1, :]
        else:
            raise AttributeError("Unable to locate internal language model in Moondream instance.")

    # 5. Process logits
    yes_logit = next_token_logits[yes_id].detach().cpu()
    no_logit  = next_token_logits[no_id].detach().cpu()

    confidences = F.softmax(torch.stack([yes_logit, no_logit]).float(), dim=0)
    yes_conf = confidences[0].item()
    no_conf  = confidences[1].item()

    # 6. Output Results
    print("-" * 40)
    print(f"Model               : {args.model_path} ({args.revision})")
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
