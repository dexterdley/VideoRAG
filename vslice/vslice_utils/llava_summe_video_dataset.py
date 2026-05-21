import os
import h5py
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_video_from_picks(video_path, picks, width=896, height=672):
    """
    Directly load the picked frames as PIL Images
    """
    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    picks_list = [min(pick, len(vr)-1) for pick in picks]
    frames = vr.get_batch(picks_list).asnumpy()
    frames = [Image.fromarray(f, mode='RGB') for f in frames]
    return frames

class SumMeLLaMA_VideoDataset(Dataset):
    def __init__(self, 
                 mode, 
                 split_idx,
                 processor,
                 clip_length=16, 
                 frame_stride=1,
                 load_test=False,
                 random_sampling=True):
        """
        SumMe Video Dataset
        """
        self.mode = mode
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.processor = processor
        self.load_test = load_test
        self.random_sampling = random_sampling
        
        self.dataset = './SumMe/eccv16_dataset_summe_google_pool5.h5'
        self.split_file = './dataset/summe_splits.json'
        self.video_folder = './SumMe/raw/videos/'
        self.video_data = h5py.File(self.dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            self.data = self.data[split_idx]
        
        self.system_prompt = "You are an expert video editor. Strictly answer only Yes or No."

    def __len__(self):
        return len(self.data[self.mode + '_keys'])

    def _sample_frame_indices(self, total_frames):
        """Sample frame indices for a clip"""
        if self.random_sampling:
            start_idx = np.random.randint(0, max(1, total_frames - self.clip_length * self.frame_stride))
        else:
            start_idx = max(0, (total_frames - self.clip_length * self.frame_stride) // 2)

        frame_indices = start_idx + np.arange(self.clip_length) * self.frame_stride
        frame_indices = np.minimum(frame_indices, total_frames - 1)
        return frame_indices.tolist()

    def __getitem__(self, index):
        video_name = self.data[self.mode + '_keys'][index]
        full_features = torch.as_tensor(self.video_data[video_name + '/features'])
        full_gtscore = torch.as_tensor(self.video_data[video_name + '/gtscore'])
        picks = self.video_data[video_name + '/picks']

        # Parse filename
        video_filename = str(np.array(self.video_data[video_name + '/video_name']))
        video_filename = video_filename.strip("b'").strip('"').strip()
        split_filename = video_filename.split(" ")
        clean_filename = "".join([item + "_" for item in split_filename])
        video_path = self.video_folder + clean_filename

        if os.path.exists(video_path + ".webm"):
            video_path += ".webm"
        else:
            video_path += ".mp4"

        # Load all picked frames as PIL Images
        video_frames = load_video_from_picks(video_path, picks)
        total_frames = len(video_frames)
        title = video_filename
        
        # You can extract keywords from your dataset if available. Defaulting here.
        keywords = "the main subject" 
        formatted_prompt = f"Does this image represent the core message of {keywords} in the video context of '{title}'?"

        if not self.load_test:
            # Training: Sample frame indices
            frame_indices = self._sample_frame_indices(total_frames)
            frames = [video_frames[i] for i in frame_indices]
            gtscore = full_gtscore[frame_indices].unsqueeze(1)
            features = full_features[frame_indices].unsqueeze(1)
        else:
            # Testing: Use all frames
            frames = video_frames
            gtscore = full_gtscore.unsqueeze(1)
            features = full_features.unsqueeze(1)

        # --- PROCESSOR LOGIC ---
        prompts_lists = []
        input_images_lists = []

        for img in frames:
            msgs = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': f"(<image>./</image>)\n{formatted_prompt}"}
            ]
            prompt_str = self.processor.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts_lists.append(prompt_str)
            input_images_lists.append([img]) # List of lists: 1 image per prompt

        # Run processor for the whole clip
        inputs = self.processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=2048
        )

        if "position_ids" not in inputs:
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        if "image_sizes" in inputs:
            inputs.pop("image_sizes")

        sample = {
            'video_name': video_name,
            'features': features,
            'gtscore': gtscore,
            'inputs': inputs,
            'title': title
        }

        # Add test-specific keys
        if self.mode != 'train':
            sample['picks'] = torch.as_tensor(np.array(picks))
            sample['n_frames'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frames']))
            sample['change_points'] = torch.as_tensor(np.array(self.video_data[video_name + '/change_points']))
            sample['n_frame_per_seg'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frame_per_seg']))
            sample['gt_summary'] = torch.as_tensor(np.array(self.video_data[video_name + '/user_summary']))

        return sample

class TrainBatchCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        # Note MiniCPM-V structure is ['input_ids', 'attention_mask', 'pixel_values', 'image_bound', 'tgt_sizes', 'position_ids']
        # [tensor, tensor, list[tensor], list[tensor], list[tensor], tensor]

    def __call__(self, batch):
        # 1. Separate your custom regression/metadata from the model inputs
        video_names = [data['video_name'] for data in batch]
        titles = [data['title'] for data in batch]
        
        # Standard pad_sequence for your regression scores and features
        frame_feat = pad_sequence([data['features'] for data in batch], batch_first=True)
        gtscore_padded = pad_sequence([data['gtscore'] for data in batch], batch_first=True)

        # 2. Extract MiniCPM's raw input dictionaries
        hf_inputs = [data['inputs'] for data in batch]
        max_len = max(x['input_ids'].size(-1) for x in hf_inputs)

        padded_input_ids = [
            F.pad(x['input_ids'], (0, max_len - x['input_ids'].size(-1)), value=self.pad_token_id)
            for x in hf_inputs
        ]
        padded_position_ids = [
            F.pad(x['position_ids'], (0, max_len - x['position_ids'].size(-1)), value=0)
            for x in hf_inputs
        ]
        padded_attention_masks = [
            F.pad(x['attention_mask'], (0, max_len - x['attention_mask'].size(-1)), value=0)
            for x in hf_inputs
        ]

        input_ids = torch.cat(padded_input_ids, dim=0)
        position_ids = torch.cat(padded_position_ids, dim=0)
        attention_mask = torch.cat(padded_attention_masks, dim=0)

        pixel_values, image_bound, tgt_sizes = [], [], []
        for x in hf_inputs:
            if 'pixel_values' in x:
                pixel_values.extend(x['pixel_values'])
            if 'image_bound' in x:
                image_bound.extend(x['image_bound'])
            if 'tgt_sizes' in x:
                 tgt_sizes.extend(x['tgt_sizes'])

        MiniCPMClass = type(hf_inputs[0])

        collated_inputs = MiniCPMClass({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'image_bound': image_bound,
            'tgt_sizes': tgt_sizes,
            'gtscore': gtscore_padded,
            'features': frame_feat,
            'video_name': video_names,
            'title': titles
        })

        return collated_inputs