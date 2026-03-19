import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import accuracy_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Pathing configuration
path_linux = '/home/dexter/VideoRAG/.checkpoints/MiniCPM-V-2_6-int4'
path_wsl = './MiniCPM-V-2_6-int4'

if os.path.exists(path_wsl):
    MODEL_PATH = path_wsl
elif os.path.exists(path_linux):
    MODEL_PATH = path_linux
else:
    MODEL_PATH = "openbmb/MiniCPM-V-2_6"
print(f"Using model path: {MODEL_PATH}")

# ─────────────────────── VLM MODEL LOADER ───────────────────────
print(f"Loading VLM Backbone: {MODEL_PATH}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"✅ VLM Model & Tokens Loaded Successfully (Yes ID: {yes_token_id}, No ID: {no_token_id}).")

except Exception as e:
    print(f"❌ Error loading VLM model: {e}")
    vlm_model, tokenizer, processor, yes_token_id, no_token_id = None, None, None, None, None

# ─────────────────────── CONFIGURATION ───────────────────────
DATA_DIR = os.path.expanduser('~/LLaVA-VLS/t3_data_files/order_3/black_bg/')
CLASSES = {'bloom': 0, 'cats': 1, 'people': 2}
CLASSES2KEYS =  {0:'bloom', 1:'cats', 2:'people'}
EVENT_DURATION_SEC = 5
FPS = 1  
N_BINS = 15
SEGMENT_LENGTH = 5 # Number of seconds (and frames) per batch

def convert_to_onehot(labels, classes=3):
    one_hot = torch.zeros(len(labels), classes, dtype=torch.float32)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return one_hot

# ─────────────────────── FUNCTIONS ───────────────────────
def generate_temporal_ground_truth(sequence_name, fps=FPS, duration=EVENT_DURATION_SEC):
    events = sequence_name.split('_')
    labels = []
    for event in events:
        if event in CLASSES:
            frames_per_event = duration * fps
            labels.extend([CLASSES[event]] * frames_per_event)
    return torch.tensor(labels, dtype=torch.long)

def get_model_predictions(video_path, num_frames, model, processor, yes_id, no_id, labels, mode):        
    # Extract frames
    vr = VideoReader(video_path, ctx=cpu(0), width=596, height=336)
    fps = vr.get_avg_fps()
    duration_sec = min(int(len(vr) / fps), len(labels))
    indices = [int(i * fps) for i in range(duration_sec)]
    batch_npy = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]

    system_prompt = "You are an expert video analyst. Answer strictly Yes or No."

    # --- Batched Inference Logic ---
    with torch.inference_mode():
        prompts_lists = []
        input_images_lists = []

        # 1. Build a mega-batch asking about EVERY class for EVERY frame
        for i, img in enumerate(frames):
            for class_id, cls_name in CLASSES2KEYS.items():
                formatted_prompt = f"Does this image contain or represent: '{cls_name}'?"
                msgs = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"(<image>./</image>)\n{formatted_prompt}"}
                ]
                prompt_str = processor.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts_lists.append(prompt_str)
                input_images_lists.append([img])
                
        # 2. Process the mega-batch
        inputs = processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=2048
        ).to(model.device)

        if "position_ids" not in inputs:
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0).expand(batch_size, -1)

        if "image_sizes" in inputs:
            inputs.pop("image_sizes")

        outputs = model(inputs, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        yes_probs = probs[:, yes_id]

        if mode == "bitemporal":
            # --- CALIBRATION: Binary Softmax (Yes vs No Competition) ---
            # Instead of softmax over 32k tokens, we isolate the two relevant outcomes.
            yes_logits = logits[:, yes_token_id]
            no_logits = logits[:, no_token_id]
            
            # Apply a mild Temperature Scaling (T=1.2) to soften the distribution
            T = 0.8
            binary_logits = torch.stack([yes_logits, no_logits], dim=-1) / T
            binary_probs = torch.nn.functional.softmax(binary_logits, dim=-1)
            
            # The calibrated confidence is the probability of 'Yes' relative only to 'No'
            yes_probs = binary_probs[:, 0]

    return yes_probs.cpu()

def compute_ece_and_bins(yes_probabilities, targets, n_bins=N_BINS):
    """
    Computes binary ECE comparing P(Yes) against actual 1/0 targets.
    """
    # Prediction is 1 (Yes) if P(Yes) >= 0.5, else 0 (No)
    predictions = (yes_probabilities >= 0.5).float()
    
    # Confidence is max(P(Yes), P(No))
    confidences = torch.where(yes_probabilities >= 0.5, yes_probabilities, 1.0 - yes_probabilities)
    
    # Compare predictions against the REAL targets (1 or 0)
    accuracies = predictions.eq(targets)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            
    return ece.item(), bin_confidences, bin_accuracies

def plot_reliability_diagram(bin_confidences, bin_accuracies, ece_score, save_path="./results/t3_reliability_diagram.png"):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    valid_confs = [c for c in bin_confidences if c > 0]
    valid_accs = [a for a, c in zip(bin_accuracies, bin_confidences) if c > 0]
    
    if valid_confs:
        plt.plot(valid_confs, valid_accs, marker='o', linewidth=2, label=f'MiniCPM (ECE = {ece_score:.4f})')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram: T3 Synthetic Video Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"\n✅ Reliability diagram saved to {save_path}")

def calibrate_bitemporal(scores, decay=0.9, sigma=1.0):
    """
    Performs bitemporal decay and Gaussian smoothing
    """
    scores = np.array(scores)
    n = len(scores)
    if n == 0:
        return scores

    forward = np.zeros(n)
    backward = np.zeros(n)

    # --- 1. Forward Pass (Bridge dropouts forward) ---
    curr_score = 0
    for i in range(n):
        curr_score = max(scores[i], curr_score * decay)
        forward[i] = curr_score

    # --- 2. Backward Pass (Bridge dropouts backward) ---
    curr_score = 0
    for i in range(n - 1, -1, -1):
        curr_score = max(scores[i], curr_score * decay)
        backward[i] = curr_score

    # --- 3. Aggregate & Smooth ---
    # Take the strongest signal from either direction to fill in gaps
    calibrated = np.maximum(forward, backward)
    
    # Apply a mild Gaussian blur to remove sharp jagged edges
    calibrated = gaussian_filter1d(calibrated, sigma=sigma)

    return np.clip(calibrated, 0, 1)

def main(args):
    if not os.path.exists(DATA_DIR):
        print(f"Warning: Directory not found at {DATA_DIR}. Running with simulated names.")
        sequences = ['bloom_cats_people']
        base_path = DATA_DIR
    else:
        folder_name = os.path.basename(os.path.normpath(DATA_DIR))
        if any(event in folder_name for event in CLASSES):
            sequences = [folder_name]
            base_path = os.path.dirname(os.path.normpath(DATA_DIR))
        else:
            sequences = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
            base_path = DATA_DIR

    all_probabilities = []
    all_labels = []
    all_ece = []

    for seq in tqdm(sequences):
        print(f"\nProcessing Sequence: {seq}")
        # 1. Get Ground Truth
        labels = generate_temporal_ground_truth(seq)
        
        # 2. Get Model Probabilities 
        video_path = os.path.join(base_path, seq)
        num_frames = len(labels)

        video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4'))]
        if video_files:
            for item in video_files:
                target_path = os.path.join(video_path, item)
                
                #print(f"Decoding path: {target_path}")

                probs = get_model_predictions(
                    video_path=target_path, 
                    num_frames=num_frames, 
                    model=vlm_model, 
                    processor=processor,
                    yes_id=yes_token_id, 
                    no_id=no_token_id,
                    labels=labels,
                    mode=args.mode
                )

                # Calculate how many frames were actually processed
                actual_frames = len(probs) // len(CLASSES)
                valid_labels = labels[:actual_frames]

                # --- APPLY TEMPORAL CALIBRATION ---
                probs_cpu = probs.cpu()
                if args.mode == "bitemporal":

                    probs_numpy = probs.view(actual_frames, len(CLASSES)).cpu().float().numpy()
                    calibrated_probs = np.zeros_like(probs_numpy)
                    
                    for c in range(len(CLASSES)):
                        calibrated_probs[:, c] = calibrate_bitemporal(probs_numpy[:, c])

                # --- NEW: FRAME-LEVEL ACCURACY & F1 (PER VIDEO) ---
                probs_reshaped_for_metrics = probs_cpu.view(actual_frames, len(CLASSES))
                frame_preds = torch.argmax(probs_reshaped_for_metrics, dim=1).numpy()
                frame_labels = valid_labels.numpy()
                
                vid_acc = accuracy_score(frame_labels, frame_preds)
                vid_f1 = f1_score(frame_labels, frame_preds, average='macro')

                vid_targets = convert_to_onehot(valid_labels, classes=len(CLASSES)).flatten()                
                vid_ece, _, _ = compute_ece_and_bins(probs_cpu, vid_targets)
                #print(f"  --> Per-Video ECE for {item}: {vid_ece*100:.4f}%")
                all_labels.append(valid_labels)
                all_probabilities.append(probs_cpu)
                all_ece.append(vid_ece)

    # Aggregate
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_targets = convert_to_onehot(all_labels, classes=len(CLASSES)).flatten()

    # 3. Measure Calibration
    ece, bin_confs, bin_accs = compute_ece_and_bins(all_probabilities, all_targets)
    
    print("-" * 40)
    print(f"Total Frames Evaluated: {len(all_labels)}")
    print(f"Overall Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Average ECE:, {sum(all_ece)/len(all_ece)*100:.4f}%")

    # 4. Plot
    save_name = f"./results/t3_reliability_{args.mode}.png"
    plot_reliability_diagram(bin_confs, bin_accs, ece, save_path=save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using direct VLM probabilities and measure calibration on T3.")
    parser.add_argument("--mode", type=str, default="regular", help="Calibration algo")
    args = parser.parse_args()
    main(args)