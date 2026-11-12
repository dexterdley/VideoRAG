"""
Model-free tests for the demo (stdlib unittest; no GPU needed).

    python -m unittest discover -s demo/tests -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEMO)

import music  # noqa: E402
from pipeline import DemoConfig, clean_text, run_pipeline  # noqa: E402
from proposals import Proposal, generate_proposals, select_clips  # noqa: E402
from render import render_vertical  # noqa: E402
from samples import describe_lora, list_lora_checkpoints, list_samples  # noqa: E402
from video_io import browser_playable, ensure_web_mp4, motion_centre_x, probe_video, sample_frames  # noqa: E402


def make_video(path, seconds=12, size="640x360", audio=True):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", path]
    subprocess.run(cmd, check=True)
    return path


def ffprobe_json(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class FakeVLM:
    """Stands in for MiniCPM: 'relevant' around t=3s and t=9s (1 fps frames, scored in chunks)."""

    def __init__(self):
        self.seen = 0
        self.lora_path, self.lora_all, self.calib, self.calls = None, False, None, []
        self.probes = set()  # (kind, question, system_prompt) sent to the Yes/No probes
        self.chats = []      # free-form generation requests (captions, title, mood)

    def set_lora(self, path=None, lora_all=None):
        self.lora_path, self.lora_all = path or None, bool(lora_all)
        self.calls.append(("set_lora", path, self.lora_all))
        return "base model (no LoRA)" if not path else f"fake adapter {os.path.basename(path)}"

    def calibrate(self, frames, question, n=12, batch_size=4, system_prompt=None):
        self.calls.append(("calibrate", n))
        self.calib = (0.5, 1.0) if n else None
        return self.calib

    def score_frames(self, frames, question, batch_size=4, system_prompt=None, progress=None):
        self.probes.add(("frame", question, system_prompt))
        t = self.seen + np.arange(len(frames), dtype=float)
        self.seen += len(frames)
        return list(0.1 + 0.8 * np.exp(-((t - 3) ** 2) / 2) + 0.6 * np.exp(-((t - 9) ** 2) / 2))

    def score_clip(self, frames, question, system_prompt=None):
        self.probes.add(("clip", question, system_prompt))
        return 0.7

    def chat(self, frames, text, max_new_tokens=48, system_prompt=None):
        self.chats.append(text)
        return "energetic" if "music" in text else '"Wow, look at this!" #fyp 🔥'


class TestProposals(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0, 60, 1.0)
        self.scores = 0.05 + 0.9 * np.exp(-((self.times - 20) ** 2) / 8) + 0.6 * np.exp(-((self.times - 45) ** 2) / 4)

    def test_peaks_found_and_bounded(self):
        props = generate_proposals(self.times, self.scores, 60.0, min_len=2, max_len=6)
        self.assertGreaterEqual(len(props), 2)
        peaks = sorted(p.peak for p in props[:2])
        self.assertAlmostEqual(peaks[0], 20, delta=1.5)
        self.assertAlmostEqual(peaks[1], 45, delta=1.5)
        for p in props:
            self.assertGreaterEqual(p.start, 0)
            self.assertLessEqual(p.end, 60)
            self.assertLessEqual(p.duration, 6 + 1e-6)
            self.assertGreaterEqual(p.duration, 2 - 1e-6)
        self.assertGreater(props[0].frame_score, props[-1].frame_score)

    def test_no_overlap_after_nms(self):
        props = generate_proposals(self.times, self.scores, 60.0, nms_iou=0.0)
        for i, a in enumerate(props):
            for b in props[i + 1:]:
                self.assertLessEqual(min(a.end, b.end) - max(a.start, b.start), 1e-6)

    def test_edge_and_flat_inputs(self):
        self.assertEqual(generate_proposals([], [], 10.0), [])
        flat = generate_proposals(np.arange(10.0), np.full(10, 0.3), 10.0, min_len=2, max_len=4)
        self.assertTrue(flat and all(0 <= p.start < p.end <= 10 for p in flat))
        edge = generate_proposals(np.arange(10.0), np.r_[1.0, np.zeros(9)], 10.0, min_len=3, max_len=4)
        self.assertEqual(edge[0].start, 0.0)
        self.assertGreaterEqual(edge[0].duration, 3 - 1e-6)

    def test_select_respects_budget_and_order(self):
        props = [Proposal(10, 16, 13, 0.9), Proposal(30, 36, 33, 0.8), Proposal(0, 6, 3, 0.7),
                 Proposal(12, 18, 15, 0.95)]
        sel = select_clips(props, target_duration=14, max_clips=5)
        self.assertAlmostEqual(sum(p.duration for p in sel), 14, places=2)
        self.assertEqual([p.start for p in sel], sorted(p.start for p in sel))
        for i, a in enumerate(sel):
            for b in sel[i + 1:]:
                self.assertLessEqual(min(a.end, b.end), max(a.start, b.start))
        # (10-16) overlaps the better (12-18) -> trimmed to its free part 10-12 (not dropped),
        # then merged with the touching 12-18 into one continuous shot.
        self.assertEqual([(p.start, p.end) for p in sel], [(10, 18), (30, 36)])
        self.assertEqual(sel[0].extra.get("merged"), 2)

    def test_select_drops_weak_and_trims_overlap(self):
        sel = select_clips([Proposal(0, 4, 2, 0.8), Proposal(10, 14, 12, 0.1)], target_duration=20)
        self.assertEqual([(p.start, p.end) for p in sel], [(0, 4)])  # 0.1 < 0.35 * 0.8
        sel = select_clips([Proposal(10, 16, 13, 0.9), Proposal(14, 20, 17, 0.8), Proposal(15, 16.5, 16, 0.7)],
                           target_duration=20)
        self.assertEqual([(p.start, p.end) for p in sel], [(10, 20)])  # 14-20 -> 16-20, merged

    def test_add_lead_in(self):
        from proposals import add_lead_in
        p = add_lead_in(Proposal(10, 16, 13, 0.9), 5)
        self.assertEqual((p.start, p.end, p.peak, p.extra["lead_in"]), (5, 16, 13, 5))
        p = add_lead_in(Proposal(2, 8, 5, 0.9), 5)           # clamped at the start of the video
        self.assertEqual((p.start, p.extra["lead_in"]), (0, 2))
        p = add_lead_in(Proposal(0, 6, 3, 0.9), 5)
        self.assertEqual((p.start, p.extra), (0, {}))

    def test_clip_score_changes_rank(self):
        a, b = Proposal(0, 4, 2, 0.6), Proposal(10, 14, 12, 0.5)
        a.clip_score, b.clip_score = 0.1, 0.9
        self.assertEqual(select_clips([a, b], target_duration=4, max_clips=1)[0].start, 10)


class TestMusicAndText(unittest.TestCase):
    def test_render_all_moods(self):
        for mood in music.MOOD_NAMES:
            a = music.render(mood, 3.7)
            self.assertEqual(a.shape, (int(3.7 * music.SR), 2))
            self.assertLessEqual(np.abs(a).max(), 1.0)
            self.assertGreater(np.abs(a).mean(), 0.01, mood)

    def test_write_wav(self):
        with tempfile.TemporaryDirectory() as d:
            p = music.write_wav(os.path.join(d, "x.wav"), music.render("chill", 2.0))
            self.assertAlmostEqual(float(ffprobe_json(p)["format"]["duration"]), 2.0, delta=0.05)

    def test_clean_text(self):
        self.assertEqual(clean_text('"Wow, look at this!" #fyp 🔥\nsecond line', 10), "Wow, look at this!")
        self.assertEqual(clean_text("one two three four five", 3), "one two three")


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.with_audio = make_video(os.path.join(cls.tmp, "a.mp4"), audio=True)
        cls.no_audio = make_video(os.path.join(cls.tmp, "b.mp4"), size="320x240", audio=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_probe_and_sample(self):
        info = probe_video(self.with_audio)
        self.assertEqual((info.width, info.height, info.has_audio), (640, 360, True))
        self.assertFalse(probe_video(self.no_audio).has_audio)
        frames, times = sample_frames(self.with_audio, sample_fps=2.0, max_side=200)
        self.assertEqual(len(frames), 25)  # 300 frames, step round(25/2)=12
        self.assertTrue(np.allclose(np.diff(times), 0.48, atol=0.05))
        self.assertLessEqual(max(frames[0].size), 200)

    def test_ensure_web_mp4(self):
        self.assertTrue(browser_playable(self.with_audio))
        self.assertEqual(ensure_web_mp4(self.with_audio, self.tmp), self.with_audio)
        # H.264 in an AVI container (like the SumMe '.mp4's) -> lossless remux to MP4
        avi = os.path.join(self.tmp, "fake.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.no_audio, "-c", "copy", "-f", "avi", avi],
                       check=True)
        self.assertFalse(browser_playable(avi))
        web = ensure_web_mp4(avi, os.path.join(self.tmp, "web"))
        self.assertTrue(browser_playable(web))
        self.assertEqual(probe_video(web).width, 320)
        # MPEG-4 Part 2 -> transcoded to H.264
        mp4v = os.path.join(self.tmp, "mp4v.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.no_audio, "-c:v", "mpeg4", mp4v], check=True)
        self.assertTrue(browser_playable(ensure_web_mp4(mp4v, os.path.join(self.tmp, "web"))))

    def test_motion_centre_in_range(self):
        frames, times = sample_frames(self.with_audio, 1.0)
        c = motion_centre_x(frames, times, 0, 12, 0.3)
        self.assertTrue(0.15 <= c <= 0.85)
        self.assertEqual(motion_centre_x(frames, times, 0, 12, 1.0), 0.5)

    def _check_vertical(self, out, W, H, expect_dur):
        meta = ffprobe_json(out)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (W, H))
        self.assertTrue(any(s["codec_type"] == "audio" for s in meta["streams"]))
        self.assertAlmostEqual(float(meta["format"]["duration"]), expect_dur, delta=0.25)

    def test_render_blur_with_music_and_text(self):
        out = os.path.join(self.tmp, "blur.mp4")
        wav = music.write_wav(os.path.join(self.tmp, "m.wav"), music.render("upbeat", 6.0))
        total = render_vertical(self.with_audio, probe_video(self.with_audio), [(1, 3.5), (7, 10.5)], out,
                                mode="blur", W=360, H=640, music_wav=wav, title="A hook, with: commas",
                                captions=["first 'clip'", "second: clip, 100%"])
        self.assertAlmostEqual(total, 6.0)
        self._check_vertical(out, 360, 640, 6.0)

    def test_render_crop_without_source_audio(self):
        out = os.path.join(self.tmp, "crop.mp4")
        render_vertical(self.no_audio, probe_video(self.no_audio), [(0, 2), (5, 8)], out,
                        mode="crop", W=360, H=640, crop_centres=[0.2, 0.9])
        self._check_vertical(out, 360, 640, 5.0)

    def test_pipeline_with_fake_vlm(self):
        out_dir = os.path.join(self.tmp, "run")
        cfg = DemoConfig(width=360, height=640, target_duration=6, mood="auto")
        res = run_pipeline(self.with_audio, "test pattern", FakeVLM(), out_dir, cfg, log=lambda *a: None)
        self.assertEqual(res["mood"], "energetic")
        self.assertTrue(res["title"].startswith("Wow"))
        self.assertTrue(all(os.path.exists(p) for p in res["outputs"].values() if p))
        self.assertTrue(any(abs(s["peak"] - 3) <= 1 for s in res["selected"]))
        self._check_vertical(res["outputs"]["video"], 360, 640, sum(s["duration"] for s in res["selected"]))


def make_adapter_dir(root, name, r=8):
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": r, "lora_alpha": 16, "target_modules": ["v_proj", "q_proj"]}, f)
    return path


class TestLoRA(unittest.TestCase):
    """Adapter discovery + the log-odds calibration maths (MiniCPM stubbed, no GPU)."""

    def test_discover_and_describe(self):
        with tempfile.TemporaryDirectory() as root:
            a = make_adapter_dir(root, "best_DPO_split0.pth")
            b = make_adapter_dir(os.path.join(root, "run2"), "ckpt", r=16)
            os.makedirs(os.path.join(root, "not_an_adapter"))
            found = list_lora_checkpoints(root)
            self.assertEqual(found, {"best_DPO_split0.pth": a, os.path.join("run2", "ckpt"): b})
            self.assertEqual(describe_lora(a), "LORA r=8, alpha=16, on q_proj/v_proj")

    def test_calibration_recovers_affine_map(self):
        import contextlib
        from vlm import MiniCPM, sigmoid

        rng = np.random.default_rng(0)
        base = rng.normal(-1.0, 1.2, 40)
        lora = (base - 2.0) / 0.13 + rng.normal(0, 0.05, 40)  # DPO-like: stretched, pushed towards "No"
        m = MiniCPM.__new__(MiniCPM)
        m.lora_path, m.lora_all, m.calib = "adapter", False, None
        state = {"adapter": True}

        class Peft:
            @contextlib.contextmanager
            def disable_adapter(self):
                state["adapter"] = False
                yield
                state["adapter"] = True

        m.peft = Peft()
        m._probe = lambda lists, q, sp=None, batch_size=None: (   # (log-odds, P(Yes)+P(No) = 1)
            np.array([(lora if state["adapter"] else base)[f[0]] for f in lists]), np.ones(len(lists)))
        a, b = m.calibrate(list(range(40)), "q", n=12)
        self.assertAlmostEqual(a, 0.13, delta=0.01)
        self.assertAlmostEqual(b, 2.0, delta=0.15)
        scores = np.array(m.score_frames(list(range(40)), "q", batch_size=8))
        self.assertLess(np.abs(scores - sigmoid(base)).max(), 0.05)       # back on the base model's scale
        self.assertEqual(list(np.argsort(scores)), list(np.argsort(lora)))  # ordering is the adapter's
        self.assertIsNone(m.calibrate(list(range(40)), "q", n=0))          # n=0 -> raw adapter scores
        self.assertLess(np.array(m.score_frames(list(range(40)), "q")).max(), 1e-3)


class TestApp(unittest.TestCase):
    """Drive the UI handlers (analyze -> pick proposals -> create) with the fake VLM."""

    @classmethod
    def setUpClass(cls):
        import app
        cls.app = app
        cls.tmp = tempfile.mkdtemp()
        cls.video = make_video(os.path.join(cls.tmp, "in.mp4"), seconds=14)
        cls.old_root, app.OUT_ROOT = app.OUT_ROOT, os.path.join(cls.tmp, "out")

    @classmethod
    def tearDownClass(cls):
        cls.app.OUT_ROOT = cls.old_root
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def analyze(self):
        vlm = FakeVLM()
        events = list(self.app.analyze_events(self.video, "test pattern", 6, 4, vlm_getter=lambda: vlm))
        merged = {}
        for ev in events:
            merged.update(ev)
        return events, merged

    def test_analyze_streams_and_builds_cards(self):
        events, final = self.analyze()
        plots = [e["plot"] for e in events if "plot" in e]
        self.assertGreater(len(plots), 1)  # live plot updated more than once
        self.assertEqual(len(plots[-1]), 2 * 14)  # per-frame + smoothed series
        progress = [e["progress"] for e in events if "progress" in e]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 100)
        cards = final["cards"]
        self.assertTrue(1 <= len(cards) <= self.app.MAX_CARDS)
        self.assertTrue(any(c["checked"] for c in cards))
        for c in cards:
            self.assertTrue(os.path.exists(c["preview"]))
            self.assertEqual(c["caption"], "Wow, look at this!")
        self.assertTrue(final["ready"] and final["run_id"] in self.app.RUNS)
        self.assertTrue(os.path.exists(final["timeline"]))

    def test_create_from_chosen_cards(self):
        _, final = self.analyze()
        n = len(final["cards"])
        includes = [i == 0 for i in range(self.app.MAX_CARDS)]
        captions = ["My edited caption"] + [""] * (self.app.MAX_CARDS - 1)
        events = list(self.app.create_events(final["run_id"], includes, captions, "Edited title", "crop",
                                             "720 x 1280 (HD, faster)", "auto", True, True, 0.3,
                                             vlm_getter=FakeVLM))
        out = events[-1]["video"]
        self.assertEqual(events[-1]["download"], out)
        self.assertIn("energetic", "".join(e.get("log", "") for e in events))
        card = self.app.RUNS[final["run_id"]]["cards"][0]
        meta = ffprobe_json(out)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (720, 1280))
        self.assertAlmostEqual(float(meta["format"]["duration"]), card.duration, delta=0.25)
        self.assertGreaterEqual(n, 1)

    def test_create_requires_selection(self):
        import gradio as gr
        _, final = self.analyze()
        with self.assertRaises(gr.Error):
            list(self.app.create_events(final["run_id"], [False] * 8, [""] * 8, "", "blur",
                                        "720 x 1280 (HD, faster)", "upbeat", False, False, 1.0))
        with self.assertRaises(gr.Error):
            list(self.app.analyze_events(None, "x", 6, 4))

    def test_ui_builds(self):
        self.assertIsNotNone(self.app.build_ui())

    def test_lead_in_extends_cards_backwards(self):
        def cards(lead_in):
            vlm = FakeVLM()
            list(self.app.analyze_events(self.video, "test pattern", 6, 4, lead_in=lead_in, vlm_getter=lambda: vlm))
            return {c.end: c for c in self.app.RUNS[list(self.app.RUNS)[-1]]["cards"]}

        core, led = cards(0), cards(4)
        self.assertEqual(set(core), set(led))  # same detections, same ends
        for end, c in core.items():
            self.assertAlmostEqual(led[end].start, max(0.0, c.start - 4), places=2)
            self.assertEqual((led[end].peak, led[end].frame_score, led[end].clip_score),
                             (c.peak, c.frame_score, c.clip_score))  # ranking/verification on the core
        self.assertTrue(any(led[e].duration > core[e].duration for e in core))

    def test_create_with_music_none(self):
        _, final = self.analyze()

        def no_vlm():
            raise AssertionError("MiniCPM should not be asked for a mood when music is none")

        events = list(self.app.create_events(final["run_id"], [True] + [False] * 7, [""] * 8, "", "blur",
                                             "720 x 1280 (HD, faster)", "none", True, False, 0.3,
                                             vlm_getter=no_vlm))
        self.assertIn("no music", "\n".join(e.get("log", "") for e in events))
        out = events[-1]["video"]
        with open(os.path.splitext(out)[0] + ".json") as f:
            meta = json.load(f)
        self.assertEqual((meta["mood"], meta["music"]), (None, None))
        self.assertTrue(any(s["codec_type"] == "audio" for s in ffprobe_json(out)["streams"]))  # original audio

    def test_analyze_naraka_prompt_like_naraka_gui(self):
        from pipeline import NARAKA_PROMPT, NARAKA_SYSTEM
        from proposals import calibrate_bitemporal
        vlm = FakeVLM()
        events = list(self.app.analyze_events(self.video, NARAKA_PROMPT, 6, 4, vlm_getter=lambda: vlm))
        self.assertEqual(vlm.probes, {("frame", NARAKA_PROMPT, NARAKA_SYSTEM), ("clip", NARAKA_PROMPT, NARAKA_SYSTEM)})
        self.assertIn("sent verbatim", events[-1]["log"])
        # Captions / title must not quote the "answer 'Yes' ... no explanation" instruction (MiniCPM obeys it).
        self.assertTrue(vlm.chats)
        for text in vlm.chats:
            self.assertNotIn("answer 'Yes'", text)
            self.assertIn('"highlight moments"', text)
        plot = [e["plot"] for e in events if "plot" in e][-1]
        raw = plot[plot.Series == "P(Yes) per frame"].Score.to_numpy()
        np.testing.assert_allclose(plot[plot.Series == "importance"].Score.to_numpy(), calibrate_bitemporal(raw))

    def test_analyze_with_lora(self):
        import gradio as gr
        with tempfile.TemporaryDirectory() as root:
            adapter = make_adapter_dir(root, "best_DPO_split0.pth")
            vlm = FakeVLM()
            events = list(self.app.analyze_events(self.video, "test pattern", 6, 4, adapter, True, True,
                                                  vlm_getter=lambda: vlm))
            final = {k: v for e in events for k, v in e.items()}
            self.assertEqual(vlm.calls[0], ("set_lora", adapter, True))
            self.assertIn(("calibrate", 12), vlm.calls)
            self.assertIn("fake adapter best_DPO_split0.pth", final["log"])
            self.assertIn("calibrated", final["log"])
            run = self.app.RUNS[final["run_id"]]
            self.assertEqual((run["lora"], run["lora_all"]), (adapter, True))
            with open(os.path.join(run["out_dir"], "proposals.json")) as f:
                self.assertEqual(json.load(f)["lora"], adapter)

            # Create re-applies the run's adapter before MiniCPM picks the mood.
            vlm2 = FakeVLM()
            list(self.app.create_events(final["run_id"], [True] + [False] * 7, [""] * 8, "", "blur",
                                        "720 x 1280 (HD, faster)", "auto", True, False, 0.3,
                                        vlm_getter=lambda: vlm2))
            self.assertEqual(vlm2.calls[0], ("set_lora", adapter, True))

            # Uncalibrated + base model choices.
            vlm3 = FakeVLM()
            list(self.app.analyze_events(self.video, "test pattern", 6, 4, adapter, False, False,
                                         vlm_getter=lambda: vlm3))
            self.assertIn(("calibrate", 0), vlm3.calls)
            vlm4 = FakeVLM()
            list(self.app.analyze_events(self.video, "test pattern", 6, 4, self.app.BASE_MODEL,
                                         vlm_getter=lambda: vlm4))
            self.assertEqual(vlm4.calls, [("set_lora", None, False)])  # no calibration without an adapter

            with self.assertRaises(gr.Error):
                list(self.app.analyze_events(self.video, "x", 6, 4, os.path.join(root, "missing")))


NARAKA_GUI = os.path.join(os.path.dirname(DEMO), "vslice", "naraka_gui.py")


def naraka_gui_defs(*names, **namespace):
    """Exec only the named top-level defs of vslice/naraka_gui.py (importing it loads the model)."""
    import ast
    if not os.path.exists(NARAKA_GUI):
        raise unittest.SkipTest("vslice/naraka_gui.py not found")
    with open(NARAKA_GUI, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), NARAKA_GUI, "exec"), namespace)
    return [namespace[n] for n in names]


class TestNarakaParity(unittest.TestCase):
    """The demo's frames, prompts, P(Yes) and importance curve are vslice/naraka_gui.py's."""

    def test_importance_curve_is_naraka_calibration(self):
        from scipy.ndimage import gaussian_filter1d
        from proposals import calibrate_bitemporal
        (ref,) = naraka_gui_defs("calibrate_bitemporal", np=np, gaussian_filter1d=gaussian_filter1d)
        rng = np.random.default_rng(1)
        sparse = np.where(rng.random(300) < 0.03, rng.uniform(0.6, 1.0, 300), rng.uniform(0, 0.05, 300))
        for x in (sparse, rng.random(50), np.zeros(10), np.array([0.7]), np.array([])):
            np.testing.assert_allclose(calibrate_bitemporal(x), ref(np.asarray(x, dtype=float)), atol=1e-12)

    def test_frame_indices_are_naraka_segments(self):
        from video_io import naraka_frame_indices

        class FakeReader:  # decord.VideoReader stand-in: get_batch(idx) "decodes" to the indices
            def __init__(self, path, ctx=None, width=None, height=None):
                self.n, self.fps = path

            def __len__(self):
                return self.n

            def get_avg_fps(self):
                return self.fps

            def get_batch(self, idx):
                return type("B", (), {"asnumpy": lambda _: np.asarray(idx)})()

        image = type("I", (), {"fromarray": staticmethod(lambda f, mode=None: int(f))})
        (Dataset,) = naraka_gui_defs("VideoSegmentDataset", VideoReader=FakeReader, cpu=lambda i: None,
                                     Dataset=object, Image=image)
        for n, fps in [(83380, 60.0), (62872, 45.0), (4200, 29.97), (2000, 30000 / 1001), (7681, 60.0),
                       (3841, 30.0), (1500, 23.976), (100, 25.0)]:
            ds = Dataset((n, fps), segment_length=64)
            ref = [i for k in range(len(ds)) for i in ds[k][0]]
            self.assertEqual(naraka_frame_indices(n, fps), ref, f"n={n} fps={fps}")

    def test_sample_frames_picks_naraka_frames(self):
        from video_io import naraka_frame_indices
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ntsc.mp4")  # 29.97 fps: naraka steps int(fps)=29, re-phased at 64 s
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=96x54:rate=30000/1001:duration=70", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", path], check=True)
            frames, times = sample_frames(path, 1.0)
            n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                                   capture_output=True, text=True, check=True).stdout)
        fps = 30000 / 1001
        self.assertEqual(list(np.round(times * fps).astype(int)), naraka_frame_indices(n, fps))
        self.assertEqual(len(frames), len(times))
        self.assertEqual(frames[0].size, (96, 54))

    def test_scoring_prompts(self):
        from pipeline import NARAKA_PROMPT, NARAKA_SYSTEM, NARAKA_SYSTEM_ZH, scoring_prompts
        self.assertEqual(scoring_prompts(NARAKA_PROMPT), (NARAKA_PROMPT, NARAKA_PROMPT, NARAKA_SYSTEM))
        zh = ("分析这些画面，检测是否存在类似 '首胜'、'2连胜'、'3连胜'、'4连胜' 或 '无敌' 的文字横幅。"
              "请忽略画面中的其他所有横幅或文字。如果存在上述任意一种，请回答 'Yes'。否则，请回答 'No'。不要提供任何解释。")
        self.assertEqual(scoring_prompts(zh), (zh, zh, NARAKA_SYSTEM_ZH))
        for desc in ("Player scoring goals and penalty kicks.", "no people, just yesterday's ramp"):
            frame_q, clip_q, system = scoring_prompts(desc)
            self.assertIn(f'"{desc}"', frame_q)
            self.assertTrue(frame_q.endswith("Answer Yes or No.") and clip_q.endswith("Answer Yes or No."))
            self.assertIsNone(system)

    def test_p_yes_is_full_softmax(self):
        """score_frames == softmax(logits)[Yes] at the last position, as in naraka_gui.py."""
        import torch
        from vlm import YES_NO_SYSTEM, MiniCPM

        logits = torch.randn(5, 3, 40) * 3
        logits[1, -1, :] = 0.0
        logits[1, -1, 7] = 6.0  # most mass on a non Yes/No token
        seen = {}

        class Batch(dict):
            def to(self, device):
                return self

        def processor(prompts, imgs, **kw):
            seen.update(prompts=prompts, kw=kw)
            return Batch(input_ids=torch.zeros(len(prompts), 3, dtype=torch.long),
                         attention_mask=torch.ones(len(prompts), 3, dtype=torch.long), image_sizes=None)

        rows = iter(range(5))
        m = MiniCPM.__new__(MiniCPM)
        m.lora_path, m.calib, m.device, m.yes_id, m.no_id = None, None, "cpu", 3, 5
        m.processor = processor
        m.tokenizer = type("T", (), {"apply_chat_template": staticmethod(
            lambda msgs, tokenize, add_generation_prompt: repr(msgs))})()
        m.model = lambda inputs, attention_mask: type("O", (), {
            "logits": torch.stack([logits[next(rows)] for _ in range(len(attention_mask))])})()

        scores = m.score_frames(["f0", "f1", "f2", "f3", "f4"], "Q?", batch_size=2, system_prompt="SYS")
        np.testing.assert_allclose(scores, torch.softmax(logits[:, -1, :], -1)[:, 3].numpy(), rtol=1e-5)
        self.assertLess(scores[1], 0.01)
        self.assertIn("'role': 'system', 'content': 'SYS'", seen["prompts"][0])
        self.assertIn("(<image>./</image>)\\nQ?", seen["prompts"][0])
        self.assertEqual((seen["kw"]["max_slice_nums"], seen["kw"]["use_image_id"]), (1, False))
        rows = iter([0])
        m.score_frames(["f0"], "Q?")
        self.assertIn(YES_NO_SYSTEM, seen["prompts"][0])  # system_prompt=None -> default editor prompt


class TestSamples(unittest.TestCase):
    def test_dataset_samples_found(self):
        s = list_samples()
        if not s:
            self.skipTest("SumMe/TVSum videos not present")
        self.assertTrue(all(os.path.exists(v["path"]) for v in s.values()))
        tv = [k for k in s if k.startswith("tvsum/")]
        if tv:
            self.assertNotEqual(s[tv[0]]["title"], tv[0].split("/")[1])  # real title from info.tsv


if __name__ == "__main__":
    unittest.main()
