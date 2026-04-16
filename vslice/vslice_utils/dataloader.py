import os
import cv2
import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

# ─────────────────────── VIDEO DATASET ───────────────────────
class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_length=32, width=896, height=672, picks=None):
        self.video_path = video_path
        self.segment_length = segment_length
        self.width, self.height = width, height

        vr = VideoReader(self.video_path, ctx=cpu(0))
        self.fps = vr.get_avg_fps()
        self.duration = len(vr) / self.fps
        num_frames = len(vr)
        del vr

        if picks is not None:
            self.picks = picks
        else:
            # Fallback: one pick per second if no picks provided
            self.picks = np.arange(0, num_frames, max(1, int(self.fps)))

        # Chunk picks into segments of size segment_length
        self.chunks = [self.picks[i : i + segment_length] for i in range(0, len(self.picks), segment_length)]

    def __len__(self): 
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        vr = VideoReader(self.video_path, ctx=cpu(0), width=self.width, height=self.height)
        indices = [int(min(p, len(vr)-1)) for p in chunk]
        batch_npy = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(f, mode='RGB') for f in batch_npy]
        
        # Calculate start and end seconds based on picks for logging
        start_sec = chunk[0] / self.fps
        end_sec = chunk[-1] / self.fps
        
        return frames, start_sec, end_sec

# ──────────────────────── DATASET BUILDERS ────────────────────────

def build_summe_manifest(root_dir):
    """
    Build a list of dicts for every SumMe video:
      {h5_key, video_name, title, video_path, gtscore, picks, n_frames}
    """
    h5_path = os.path.join(root_dir, "SumMe", "eccv16_dataset_summe_google_pool5.h5")
    video_dir = os.path.join(root_dir, "SumMe", "raw", "videos")

    f = h5py.File(h5_path, "r")
    manifest = []

    # Build a filename lookup (strip trailing underscore and extension)
    video_files = {}
    if os.path.isdir(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(".webm"):
                # SumMe naming: "Air_Force_One_.mp4" -> key "Air_Force_One"
                clean = fname.replace(".webm", "").rstrip("_")
                video_files[clean] = os.path.join(video_dir, fname)
                # Also try with spaces replaced by underscores
                video_files[clean.replace(" ", "_")] = os.path.join(video_dir, fname)

    for key in sorted(f.keys()):
        grp = f[key]
        raw = grp["video_name"][...]
        if hasattr(raw, "item"):
            raw = raw.item()
        vname = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        vname = vname.strip()

        # Try to find the raw video
        vpath = video_files.get(vname) or video_files.get(vname.replace(" ", "_"))

        if vpath is None:
            print(f"  [WARN] SumMe video not found for '{vname}', skipping")
            continue

        # The title is the video name with underscores → spaces
        title = vname.replace("_", " ").strip()

        manifest.append({
            "h5_key": key,
            "dataset": "summe",
            "video_name": vname,
            "title": title,
            "video_path": vpath,
            "gtscore": grp["gtscore"][...].astype(np.float32),
            "picks": grp["picks"][...].astype(np.int64),
            "n_frames": int(grp["n_frames"][...]),
        })

    f.close()
    print(f"[DATA] SumMe: {len(manifest)} videos discovered")
    return manifest


def build_tvsum_manifest(root_dir):
    """
    Build a list of dicts for every TVSum video:
      {h5_key, video_name, title, video_path, gtscore, picks, n_frames}
    """
    h5_path = os.path.join(root_dir, "TVSum", "eccv16_dataset_tvsum_google_pool5.h5")
    video_dir = os.path.join(root_dir, "TVSum", "tvsum50_ver_1_1",
                             "ydata-tvsum50-v1_1", "video")
    info_path = os.path.join(root_dir, "TVSum", "tvsum50_ver_1_1",
                             "ydata-tvsum50-v1_1", "data", "ydata-tvsum50-info.tsv")

    # Load video titles from the info TSV
    title_map = {}
    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path, sep="\t")
        for _, row in info_df.iterrows():
            title_map[row["video_id"]] = row["title"]

    # Build a filename lookup
    video_files = {}
    if os.path.isdir(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(".mp4"):
                vid = fname.replace(".mp4", "")
                video_files[vid] = os.path.join(video_dir, fname)

    f = h5py.File(h5_path, "r")
    manifest = []

    for key in sorted(f.keys()):
        grp = f[key]
        picks = grp["picks"][...].astype(np.int64)
        gtscore = grp["gtscore"][...].astype(np.float32)
        n_frames = int(grp["n_frames"][...])

        # TVSum H5 doesn't store video_name — we need to match by index
        # The H5 keys are video_1 .. video_50, sorted alphabetically by video_id
        # We'll try to find the video by matching feature count or brute force
        # Alternatively, we store the H5 key and match later
        # 
        # Actually, let's just iterate video_files and match by checking
        # if the number of picks matches
        manifest.append({
            "h5_key": key,
            "dataset": "tvsum",
            "video_name": key,  # placeholder, resolved below
            "title": key,       # placeholder
            "video_path": None, # placeholder
            "gtscore": gtscore,
            "picks": picks,
            "n_frames": n_frames,
        })

    f.close()

    # ── Resolve TVSum H5 keys to actual video IDs ──
    # The ECCV16 H5 has 50 entries (video_1 .. video_50).
    # The alphabetical sorting of video IDs is not reliable since they have varying lengths.
    # We map robustly by comparing the number of frames.
    
    mp4_lengths = {}
    for vid, vpath in video_files.items():
        try:
            # cv2 is often faster for frame count
            cap = cv2.VideoCapture(vpath)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if num_frames <= 0:
                vr = VideoReader(vpath, ctx=cpu(0))
                num_frames = len(vr)
                del vr
            mp4_lengths[vid] = num_frames
        except Exception as e:
            print(f"  [WARN] Failed to read {vpath}: {e}")
            continue

    # Map by length with a small tolerance for codec differences
    for item in manifest:
        matched_vid = None
        for vid, length in mp4_lengths.items():
            if abs(length - item["n_frames"]) <= 2:
                matched_vid = vid
                break
        
        if matched_vid:
            item["video_name"] = matched_vid
            item["video_path"] = video_files[matched_vid]
            item["title"] = title_map.get(matched_vid, matched_vid)
            # Remove from mp4_lengths to prevent double mapping
            del mp4_lengths[matched_vid]
        else:
            print(f"  [WARN] Could not find MP4 match for {item['h5_key']} with {item['n_frames']} frames.")

    # Filter out unresolved entries
    resolved = [m for m in manifest if m["video_path"] is not None]
    print(f"[DATA] TVSum: {len(resolved)} / {len(manifest)} videos resolved")
    return resolved