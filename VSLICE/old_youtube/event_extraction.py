"""
Event Extraction — extract discrete events from a continuous score curve.

Given per-frame engagement scores s(t), extract contiguous high-importance
regions as events with start/end timestamps and importance scores.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def extract_events(scores, times, min_duration=5.0, merge_gap=3.0, 
                   sigma=2.0, threshold_k=0.75):
    """
    Extract events from a continuous engagement score curve.
    
    Args:
        scores: np.array of shape [T], engagement scores in [0, 1]
        times: np.array of shape [T], timestamps in seconds
        min_duration: minimum event duration in seconds
        merge_gap: merge events closer than this (seconds)
        sigma: Gaussian smoothing sigma (seconds)
        threshold_k: threshold = mean + k * std
    
    Returns:
        List of dicts with keys: start, end, score (AUC), peak_score
    """
    scores = np.asarray(scores, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    
    # 1. Smooth
    smoothed = gaussian_filter1d(scores, sigma=sigma)
    
    # 2. Adaptive threshold
    threshold = np.mean(smoothed) + threshold_k * np.std(smoothed)
    threshold = max(threshold, 0.1)  # floor to avoid extracting everything
    
    # 3. Find contiguous above-threshold regions
    above = smoothed > threshold
    events = []
    
    if not np.any(above):
        return events
    
    # Find region boundaries
    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases  
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(above)]])
    
    for s_idx, e_idx in zip(starts, ends):
        start_time = times[s_idx]
        end_time = times[min(e_idx, len(times) - 1)]
        duration = end_time - start_time
        
        if duration >= min_duration:
            region_scores = smoothed[s_idx:e_idx]
            region_times = times[s_idx:e_idx]
            auc = float(np.trapezoid(region_scores, region_times))
            peak = float(np.max(region_scores))
            events.append({
                "start": float(start_time),
                "end": float(end_time),
                "score": auc,
                "peak_score": peak,
            })
    
    # 4. Merge nearby events
    events = _merge_nearby(events, gap=merge_gap)
    
    # 5. Re-compute AUC after merging and rank
    for ev in events:
        mask = (times >= ev["start"]) & (times <= ev["end"])
        if np.any(mask):
            ev["score"] = float(np.trapezoid(smoothed[mask], times[mask]))
            ev["peak_score"] = float(np.max(smoothed[mask]))
    
    events.sort(key=lambda e: e["score"], reverse=True)
    return events


def _merge_nearby(events, gap=3.0):
    """Merge events that are closer than `gap` seconds."""
    if len(events) <= 1:
        return events
    
    events = sorted(events, key=lambda e: e["start"])
    merged = [events[0]]
    
    for ev in events[1:]:
        prev = merged[-1]
        if ev["start"] - prev["end"] <= gap:
            # Merge
            prev["end"] = max(prev["end"], ev["end"])
            prev["score"] = prev["score"] + ev["score"]
            prev["peak_score"] = max(prev["peak_score"], ev["peak_score"])
        else:
            merged.append(ev)
    
    return merged


def events_to_timestamps(events, fmt="mm:ss"):
    """Convert events to human-readable timestamp strings."""
    result = []
    for ev in events:
        s = ev["start"]
        e = ev["end"]
        if fmt == "mm:ss":
            start_str = f"{int(s // 60)}:{int(s % 60):02d}"
            end_str = f"{int(e // 60)}:{int(e % 60):02d}"
        else:
            start_str = f"{s:.1f}"
            end_str = f"{e:.1f}"
        result.append({
            "start_timestamp": start_str,
            "end_timestamp": end_str,
            "score": ev["score"],
            "peak_score": ev["peak_score"],
            "duration": e - s,
        })
    return result
