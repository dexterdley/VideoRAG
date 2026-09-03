import os
import h5py
import json
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)

def load_video_from_picks(video_path, picks, width=896, height=672):
    """
    Directly load the picked frames as PIL Images
    """
    vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    picks_list = [min(pick, len(vr)-1) for pick in picks]
    frames = vr.get_batch(picks_list).asnumpy()
    frames = [Image.fromarray(f, mode='RGB') for f in frames]
    return frames

class TVSumLLaMA_VideoDataset(Dataset):

    def __init__(self, 
                mode, 
                split_idx, 
                processor, 
                clip_length=16, 
                frame_stride=1,
                load_test=False,
                random_sampling=True,
                override_keys=None):
        """
        TVSum Video Dataset.
        override_keys: if provided, use this list of H5 keys instead of the split-based key list.
                       Useful for extracting features for ALL videos (train + test) in one pass.
        """

        self.mode = mode
        self.clip_length = clip_length
        self.dataset = './TVSum/eccv16_dataset_tvsum_google_pool5.h5'
        self.split_file = './dataset/tvsum_splits.json'
        self.video_folder = './TVSum/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/'
        self.video_data = h5py.File(self.dataset, 'r')

        self.info_file = './TVSum/tvsum50_ver_1_1/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv'
        self.info_file = pd.read_csv(self.info_file, sep='\t')
        self.processor = processor
        self.frame_stride = frame_stride
        self.load_test = load_test
        self.random_sampling = random_sampling

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            self.data = self.data[split_idx]

        # If override_keys is given, use it instead of the split-derived key list
        self._keys = override_keys if override_keys is not None else self.data[self.mode + '_keys']

        self.system_prompt = "You are an expert video editor. Strictly answer only Yes or No."

    def __len__(self):
        """ Function for the 'len' operator of dataset """
        return len(self._keys)

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
        video_name = self._keys[index]
        full_features = torch.as_tensor(self.video_data[video_name + '/features'])
        full_gtscore = torch.as_tensor(self.video_data[video_name + '/gtscore'])
        picks = self.video_data[video_name + '/picks']

        video_num = int(video_name.split('_')[-1])
        video_id = self.info_file.iloc[video_num - 1]['video_id']
        video_path = self.video_folder + video_id + ".mp4"
        title = str(self.info_file.iloc[video_num - 1]['title'])

        # Load all picked frames as PIL Images
        video_frames = load_video_from_picks(video_path, picks)
        total_frames = len(video_frames)
        
        formatted_prompt = f"Does this image show a key highlight from the video titled '{title}'?"

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

class ValBatchCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __call__(self, batch):
        video_names = [data['video_name'] for data in batch]
        titles = [data['title'] for data in batch]
        
        frame_feat = pad_sequence([data['features'] for data in batch], batch_first=True)
        gtscore_padded = pad_sequence([data['gtscore'] for data in batch], batch_first=True)

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

        collated_inputs['n_frames'] = [data['n_frames'] for data in batch]
        collated_inputs['n_frame_per_seg'] = [data['n_frame_per_seg'] for data in batch]
        collated_inputs['picks'] = [data['picks'] for data in batch]
        collated_inputs['change_points'] = [data['change_points'] for data in batch]
        collated_inputs['gt_summary'] = [data['gt_summary'] for data in batch]
        return collated_inputs

class TVSumLLaMA_DPODataset(Dataset):
    def __init__(self,
                 split_idx,
                 processor,
                 clip_length=16,
                 frame_stride=1,
                 load_test=False,
                 random_sampling=True,
                 min_margin=0.2):
        """
        TVSum Video Dataset strictly for Direct Preference Optimization (DPO) Training.
        Generates pairs of Chosen (Peak) and Rejected (Valley) frames.
        """
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.processor = processor
        self.load_test = load_test
        self.random_sampling = random_sampling
        self.min_margin = min_margin
        
        self.dataset = './TVSum/eccv16_dataset_tvsum_google_pool5.h5'
        self.split_file = './dataset/tvsum_splits.json'
        self.video_folder = './TVSum/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/'
        self.video_data = h5py.File(self.dataset, 'r')
        self.epsilon = 1e-5
        self.quantile = 0.25

        info_file = './TVSum/tvsum50_ver_1_1/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv'
        self.info_file = pd.read_csv(info_file, sep='\t')

        with open(self.split_file, 'r') as f:
            data = json.loads(f.read())
            self.train_keys = data[split_idx]['train_keys']

        self.system_prompt = "You are an expert video editor. Strictly answer only Yes or No."

        # Build sub-clips per video
        self.video_pools = []
        for video_name in self.train_keys:
            gtscore = np.array(self.video_data[video_name + '/gtscore'])
            vid_len = len(gtscore)
            sub_clips = []
            for start_idx in range(0, vid_len - self.clip_length, self.clip_length):
                end_idx = start_idx + self.clip_length
                sub_scores = np.mean(gtscore[start_idx:end_idx])
                sub_clips.append((video_name, start_idx, end_idx, sub_scores))
            sub_clips.sort(key=lambda x: x[3], reverse=True)
            k = max(1, int(len(sub_clips) * self.quantile))
            chosen = sub_clips[:k]   # peaks
            rejected = sub_clips[-k:] # valleys
            valid_chosen = [c for c in chosen if c[3] - rejected[0][3] >= self.min_margin]
            if valid_chosen:
                self.video_pools.append((valid_chosen, rejected))
        # Build initial paired pool
        self.preference_pools = []
        self.shuffle_preference_pools()

    def shuffle_preference_pools(self):
        """Re-pairs chosen and rejected clips randomly for a new epoch."""
        self.preference_pools = []
        for valid_chosen, rejected in self.video_pools:
            # Shuffle rejected candidate order to create new pairings
            shuffled_rejected = list(rejected)
            random.shuffle(shuffled_rejected)
            
            for c in valid_chosen:
                r = random.choice(shuffled_rejected)
                self.preference_pools.append((c, r))

    def __len__(self):
        return len(self.preference_pools)

    def _process_clip(self, frames, formatted_prompt):
        """Helper function to run the VLM processor over a list of PIL frames"""
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
            input_images_lists.append([img])

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

        return inputs

    def __getitem__(self, index):
        chosen_clip, rejected_clip = self.preference_pools[index]

        video_name, chosen_start_idx, chosen_end_idx, chosen_sub_score = chosen_clip
        _, rejected_start_idx, rejected_end_idx, rejected_sub_score = rejected_clip

        chosen_idx = range(chosen_start_idx, chosen_end_idx)
        rejected_idx = range(rejected_start_idx, rejected_end_idx)

        full_gtscore = torch.as_tensor(self.video_data[video_name + '/gtscore'])
        full_features = torch.as_tensor(self.video_data[video_name + '/features'], dtype=torch.float32)
        picks = np.array(self.video_data[video_name + '/picks'])

        # Resolve video path and title via TVSum info file
        video_num = int(video_name.split('_')[-1])
        video_id = self.info_file.iloc[video_num - 1]['video_id']
        video_path = self.video_folder + video_id + ".mp4"
        title = str(self.info_file.iloc[video_num - 1]['title'])
        title = " ".join(title.split()[:5]) # truncate due to GPU constraints

        video_frames = load_video_from_picks(video_path, picks)
        total_frames = len(video_frames)
        formatted_prompt = f"Does this image show a key highlight from the video titled '{title}'?"

        chosen_frames = [video_frames[i] for i in chosen_idx]
        rejected_frames = [video_frames[i] for i in rejected_idx]

        chosen_inputs = self._process_clip(chosen_frames, formatted_prompt)
        rejected_inputs = self._process_clip(rejected_frames, formatted_prompt)

        chosen_score, rejected_score = full_gtscore[chosen_idx], full_gtscore[rejected_idx]
        log_margin = torch.log(chosen_score + self.epsilon) - torch.log(rejected_score + self.epsilon)

        return {
            'video_name': video_name,
            'title': title,
            'chosen_inputs': chosen_inputs,
            'rejected_inputs': rejected_inputs,
            'chosen_gt': chosen_score,
            'rejected_gt': rejected_score,
            'log_margin': log_margin,
            'chosen_idx':  chosen_idx,
            'rejected_idx': rejected_idx,
            'features': full_features,
            'picks': torch.tensor(picks, dtype=torch.long)
        }

class DPOTrainBatchCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def _collate_hf_inputs(self, hf_inputs):
        """Helper to pad and stack MiniCPM input dictionaries"""
        max_len = max(x['input_ids'].size(-1) for x in hf_inputs)

        padded_input_ids = [F.pad(x['input_ids'], (0, max_len - x['input_ids'].size(-1)), value=self.pad_token_id) for x in hf_inputs]
        padded_position_ids = [F.pad(x['position_ids'], (0, max_len - x['position_ids'].size(-1)), value=0) for x in hf_inputs]
        padded_attention_masks = [F.pad(x['attention_mask'], (0, max_len - x['attention_mask'].size(-1)), value=0) for x in hf_inputs]

        input_ids = torch.cat(padded_input_ids, dim=0)
        position_ids = torch.cat(padded_position_ids, dim=0)
        attention_mask = torch.cat(padded_attention_masks, dim=0)

        pixel_values, image_bound, tgt_sizes = [], [], []
        for x in hf_inputs:
            if 'pixel_values' in x: pixel_values.extend(x['pixel_values'])
            if 'image_bound' in x: image_bound.extend(x['image_bound'])
            if 'tgt_sizes' in x: tgt_sizes.extend(x['tgt_sizes'])

        MiniCPMClass = type(hf_inputs[0])
        return MiniCPMClass({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'image_bound': image_bound,
            'tgt_sizes': tgt_sizes,
        })

    def __call__(self, batch):
        video_names = [data['video_name'] for data in batch]
        titles = [data['title'] for data in batch]
        log_margins = torch.stack([data['log_margin'] for data in batch])
        chosen_gt = torch.stack([data['chosen_gt'] for data in batch])
        rejected_gt = torch.stack([data['rejected_gt'] for data in batch])

        chosen_idx = [data['chosen_idx'] for data in batch]
        rejected_idx = [data['rejected_idx'] for data in batch]
        features = [data['features'] for data in batch]
        picks = [data['picks'] for data in batch]

        # Collate chosen and rejected separately so your DPO loop can handle them cleanly
        chosen_inputs = self._collate_hf_inputs([data['chosen_inputs'] for data in batch])
        rejected_inputs = self._collate_hf_inputs([data['rejected_inputs'] for data in batch])

        return {
            'video_name': video_names,
            'title': titles,
            'chosen_inputs': chosen_inputs,
            'rejected_inputs': rejected_inputs,
            'chosen_gt': chosen_gt,
            'rejected_gt': rejected_gt,
            'log_margin': log_margins,
            'chosen_idx': chosen_idx,
            'rejected_idx': rejected_idx,
            'features': features,
            'picks': picks
        }