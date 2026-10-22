"""
Assemble selected clips into a vertical (9:16) short with ffmpeg, in one pass:

    per clip : trim -> reframe to WxH (blur-pad or motion-aware crop) -> fps/format
    all      : concat -> title + per-clip caption overlays -> fade in/out
    audio    : original audio (ducked) + generated BGM -> limiter -> AAC
"""

import os
import shutil
import subprocess
import tempfile
import textwrap

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def find_font():
    return next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)


def _even(x):
    return max(2, int(x) // 2 * 2)


def reframe_filter(info, mode, W, H, centre_x=0.5, tag=""):
    """ffmpeg filter chain that turns the source into exactly WxH (`tag` keeps inner labels unique)."""
    if mode == "crop":
        src_ar, dst_ar = info.width / info.height, W / H
        if src_ar > dst_ar:  # wider than 9:16 -> crop width, slide window to the action
            cw = _even(info.height * dst_ar)
            x = int(min(max(centre_x * info.width - cw / 2, 0), info.width - cw))
            crop = f"crop={cw}:{_even(info.height)}:{x}:0"
        else:                # taller than 9:16 -> crop height, centred
            ch = _even(info.width / dst_ar)
            crop = f"crop={_even(info.width)}:{ch}:0:{(info.height - ch) // 2}"
        return f"{crop},scale={W}:{H}:flags=lanczos,setsar=1"

    # "blur": whole frame letter-boxed on top of a blurred, zoomed copy of itself.
    w4, h4 = _even(W / 4), _even(H / 4)
    return (
        f"split=2[bgsrc{tag}][fgsrc{tag}];"
        f"[bgsrc{tag}]scale={w4}:{h4}:force_original_aspect_ratio=increase,crop={w4}:{h4},"
        f"boxblur=8:2,scale={W}:{H},eq=brightness=-0.08:saturation=1.2[bg{tag}];"
        f"[fgsrc{tag}]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos[fg{tag}];"
        f"[bg{tag}][fg{tag}]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def _drawtext(font, textfile, size, y, enable=None):
    expr = (f"drawtext=fontfile={font}:textfile={textfile}:fontsize={size}:fontcolor=white:"
            f"line_spacing=8:box=1:boxcolor=black@0.45:boxborderw=18:"
            f"x=(w-text_w)/2:y={y}:text_align=C")
    if enable:
        expr += f":enable='{enable}'"
    return expr


def cut_preview(video_path, start, end, out_path, height=360):
    """Small browser-friendly H.264 preview of [start, end] in the original aspect ratio."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", video_path, "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg preview failed: {proc.stderr[-1500:]}")
    return out_path


def render_vertical(video_path, info, clips, out_path, mode="blur", W=1080, H=1920, fps=30,
                    music_wav=None, orig_volume=0.35, music_volume=0.8,
                    title=None, captions=None, crop_centres=None, crf=20):
    """
    Args:
        clips:        list of (start_sec, end_sec) in output order.
        captions:     optional list[str] (same length as clips) shown at the bottom.
        crop_centres: optional list[float] horizontal crop centres (0..1) for mode='crop'.
    Returns the final duration in seconds.
    """
    if not clips:
        raise ValueError("No clips to render.")
    font = find_font()
    # Text files / filter script live in a plain temp dir: the repo path contains
    # spaces, commas and full-width parens, which ffmpeg filter args choke on.
    tmp = tempfile.mkdtemp(prefix="vslice_demo_")
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        durs = [round(e - s, 3) for s, e in clips]
        for s, d in zip((c[0] for c in clips), durs):
            cmd += ["-ss", f"{s:.3f}", "-t", f"{d:.3f}", "-i", video_path]
        if music_wav:
            cmd += ["-i", music_wav]
        n, total = len(clips), sum(durs)

        graph, concat_in = [], ""
        for i, d in enumerate(durs):
            cx = crop_centres[i] if crop_centres else 0.5
            graph.append(
                f"[{i}:v]trim=duration={d},setpts=PTS-STARTPTS,"
                f"{reframe_filter(info, mode, W, H, cx, tag=str(i))},fps={fps},format=yuv420p[v{i}]"
            )
            a_src = f"[{i}:a]" if info.has_audio else "anullsrc=r=44100:cl=stereo,"
            fade_out = max(0.0, d - 0.05)
            graph.append(
                f"{a_src}atrim=duration={d},asetpts=PTS-STARTPTS,aresample=44100,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,apad=whole_dur={d},"
                f"afade=t=in:d=0.05,afade=t=out:st={fade_out:.3f}:d=0.05[a{i}]"
            )
            concat_in += f"[v{i}][a{i}]"
        graph.append(f"{concat_in}concat=n={n}:v=1:a=1[vcat][acat]")

        # Text overlays + global fade.
        filters = []
        if font and title:
            tf = os.path.join(tmp, "title.txt")
            with open(tf, "w", encoding="utf-8") as f:
                f.write(textwrap.fill(title, 22))
            filters.append(_drawtext(font, tf, W // 16, int(H * 0.09)))
        if font and captions:
            t = 0.0
            for i, (cap, d) in enumerate(zip(captions, durs)):
                if cap:
                    cf = os.path.join(tmp, f"cap{i}.txt")
                    with open(cf, "w", encoding="utf-8") as f:
                        f.write(textwrap.fill(cap, 30))
                    filters.append(_drawtext(font, cf, W // 24, int(H * 0.76),
                                             enable=f"between(t,{t:.3f},{t + d:.3f})"))
                t += d
        filters.append(f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(0, total - 0.5):.3f}:d=0.5")
        graph.append("[vcat]" + ",".join(filters) + "[vout]")

        if music_wav:
            graph.append(f"[acat]volume={orig_volume}[ao]")
            graph.append(f"[{n}:a]atrim=duration={total:.3f},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                         f"aresample=44100,volume={music_volume}[am]")
            # duration=first ends early on some inputs; mix to the longest, then trim to the edit length.
            graph.append(f"[ao][am]amix=inputs=2:duration=longest:normalize=0,atrim=duration={total:.3f},"
                         "alimiter=limit=0.9[aout]")
        else:
            graph.append("[acat]alimiter=limit=0.9[aout]")

        script = os.path.join(tmp, "graph.txt")
        with open(script, "w", encoding="utf-8") as f:
            f.write(";\n".join(graph))

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cmd += ["-filter_complex_script", script, "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-3000:]}\nGraph:\n" + ";\n".join(graph))
        return total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
