"""
Turn per-frame relevance scores into ranked, non-overlapping clip proposals.

    scores --gaussian smooth--> peaks --grow while high--> windows
           --clamp [min_len, max_len]--> temporal NMS --> ranked proposals
"""

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Proposal:
    start: float
    end: float
    peak: float              # time (s) of the score peak inside the clip
    frame_score: float       # mean raw P(Yes) over the clip's sampled frames
    clip_score: float = None  # multi-frame MiniCPM verification, filled later
    caption: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def rank_score(self):
        if self.clip_score is None:
            return self.frame_score
        return 0.5 * self.frame_score + 0.5 * self.clip_score

    def to_dict(self):
        d = asdict(self)
        d.update(duration=round(self.duration, 2), rank_score=round(self.rank_score, 4))
        return d


def gaussian_smooth(x, sigma):
    """1-D gaussian smoothing with edge padding; sigma in samples."""
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    radius = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), k, mode="valid")


def temporal_iou(a, b):
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def generate_proposals(times, scores, duration, min_len=2.0, max_len=6.0,
                       sigma_sec=1.5, rel_drop=0.35, nms_iou=0.2, max_proposals=12):
    """
    Args:
        times:    (N,) sample timestamps in seconds (uniformly spaced).
        scores:   (N,) raw per-frame P(Yes).
        duration: video duration in seconds (clips are clamped to it).
        rel_drop: a clip grows outward from a peak while the normalised smoothed
                  score stays above (1 - rel_drop) * peak.
    Returns:
        list[Proposal] sorted by frame_score (best first).
    """
    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(times) == 0:
        return []
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0

    smooth = gaussian_smooth(scores, sigma_sec / dt)
    lo, hi = smooth.min(), smooth.max()
    norm = (smooth - lo) / (hi - lo) if hi - lo > 1e-6 else np.ones_like(smooth)

    # Local maxima (plateaus count once), strongest first.
    peaks = [i for i in range(len(norm))
             if (i == 0 or norm[i] > norm[i - 1]) and (i == len(norm) - 1 or norm[i] >= norm[i + 1])]
    if not peaks:
        peaks = [int(np.argmax(norm))]
    peaks.sort(key=lambda i: -norm[i])

    candidates = []
    for p in peaks:
        thr = norm[p] * (1.0 - rel_drop)
        l = p
        while l > 0 and norm[l - 1] >= thr:
            l -= 1
        r = p
        while r < len(norm) - 1 and norm[r + 1] >= thr:
            r += 1
        start, end = times[l], times[r] + dt

        # Clamp length around the peak.
        centre = times[p] + dt / 2
        if end - start > max_len:
            start = max(start, centre - max_len / 2)
            end = start + max_len
        if end - start < min_len:
            pad = (min_len - (end - start)) / 2
            start, end = start - pad, end + pad
        # Shift back inside the video, keeping the length when possible.
        if start < 0:
            end, start = min(duration, end - start), 0.0
        if end > duration:
            start, end = max(0.0, start - (end - duration)), duration

        inside = (times >= start) & (times < end)
        fs = float(scores[inside].mean()) if inside.any() else float(scores[p])
        candidates.append(Proposal(start=round(float(start), 2), end=round(float(end), 2),
                                   peak=round(float(times[p]), 2), frame_score=fs))

    # Temporal NMS on frame score.
    candidates.sort(key=lambda c: -c.frame_score)
    kept = []
    for c in candidates:
        if all(temporal_iou(c, k) <= nms_iou for k in kept):
            kept.append(c)
        if len(kept) >= max_proposals:
            break
    return kept


def _without_overlap(p, chosen, min_len):
    """Cut the parts of `p` already covered by `chosen`; keep the longest piece >= min_len (or None)."""
    pieces = [(p.start, p.end)]
    for c in chosen:
        pieces = [seg for s, e in pieces
                  for seg in ((s, min(e, c.start)), (max(s, c.end), e)) if seg[1] - seg[0] > 1e-6]
    if not pieces:
        return None
    s, e = max(pieces, key=lambda seg: seg[1] - seg[0])
    if e - s < min_len:
        return None
    if (s, e) == (p.start, p.end):
        return p
    return Proposal(start=s, end=e, peak=min(max(p.peak, s), e), frame_score=p.frame_score,
                    clip_score=p.clip_score, caption=p.caption, extra=dict(p.extra, trimmed=True))


def select_clips(proposals, target_duration=20.0, max_clips=6, min_len=1.5, min_rel_score=0.35):
    """
    Greedily take the best-ranked proposals until the target duration is reached,
    then return them in chronological order so the edit tells the story in order.

    * proposals scoring < min_rel_score * best are dropped: a shorter, on-topic
      edit beats padding to the target with irrelevant footage.
    * overlap with already-chosen clips is trimmed away (not the whole proposal).
    * the last clip is trimmed around its peak to land on the target duration.
    * clips that end up touching are merged into one continuous shot.
    """
    ranked = sorted(proposals, key=lambda p: -p.rank_score)
    if not ranked:
        return []
    floor = min_rel_score * ranked[0].rank_score
    chosen, total = [], 0.0
    for p in ranked:
        if len(chosen) >= max_clips or total >= target_duration - 1e-6:
            break
        if p.rank_score < floor:
            break
        p = _without_overlap(p, chosen, min_len)
        if p is None:
            continue
        remaining = target_duration - total
        if p.duration > remaining:
            if remaining < min_len:
                break
            # Trim around the peak so the key moment survives.
            start = min(max(p.start, p.peak - remaining / 2), p.end - remaining)
            p = Proposal(start=round(start, 2), end=round(start + remaining, 2), peak=p.peak,
                         frame_score=p.frame_score, clip_score=p.clip_score,
                         caption=p.caption, extra=dict(p.extra, trimmed=True))
        chosen.append(p)
        total += p.duration
    return merge_touching(sorted(chosen, key=lambda p: p.start))


def merge_touching(clips, gap=0.5):
    """Fuse chronologically sorted clips that (nearly) touch into one continuous shot."""
    merged = []
    for c in clips:
        prev = merged[-1] if merged else None
        if prev is None or c.start - prev.end > gap:
            merged.append(c)
            continue
        best = prev if prev.rank_score >= c.rank_score else c
        w0, w1 = prev.duration, c.duration
        merged[-1] = Proposal(
            start=prev.start, end=c.end, peak=best.peak,
            frame_score=(prev.frame_score * w0 + c.frame_score * w1) / (w0 + w1),
            clip_score=best.clip_score, caption=best.caption,
            extra=dict(best.extra, merged=prev.extra.get("merged", 1) + c.extra.get("merged", 1)))
    return merged
