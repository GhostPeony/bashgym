from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "output" / "bashgym-launch-picture.mp4"
FRAMES_DIR = ROOT / "output" / "frames"
W, H = 1920, 1080
FPS = 30
DURATION = 72.0

CREAM = "#f4f1eb"
PAPER = "#fffdfa"
INK = "#1e1e24"
INK_2 = "#29272c"
LILAC = "#a79ac7"
LILAC_LIGHT = "#d7cfea"
LILAC_DARK = "#67558f"
PEACH = "#e8a58d"
MINT = "#8fb7ad"
GOLD = "#d3a54a"
MUTED = "#777181"

GEORGIA = Path("C:/Windows/Fonts/georgia.ttf")
GEORGIA_BOLD = Path("C:/Windows/Fonts/georgiab.ttf")
MONO = Path("C:/Windows/Fonts/CascadiaMono.ttf")
MONO_BOLD = Path("C:/Windows/Fonts/CascadiaCode.ttf")


@lru_cache(maxsize=64)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    path = {
        "serif": GEORGIA,
        "serif-bold": GEORGIA_BOLD,
        "mono": MONO,
        "mono-bold": MONO_BOLD,
    }[kind]
    return ImageFont.truetype(str(path), size=size)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def ease(v: float) -> float:
    v = clamp(v)
    return 1 - (1 - v) ** 3


def ease_in_out(v: float) -> float:
    v = clamp(v)
    return v * v * (3 - 2 * v)


def pop(v: float) -> float:
    v = clamp(v)
    return 1 + 0.16 * math.sin(v * math.pi) * (1 - v)


def alpha_image(image: Image.Image, alpha: float) -> Image.Image:
    image = image.copy().convert("RGBA")
    image.putalpha(image.getchannel("A").point(lambda p: int(p * clamp(alpha))))
    return image


def load(name: str) -> Image.Image:
    return Image.open(ASSETS / name).convert("RGBA")


PEONY = load("bashgym-peony.png")
CURRENT_WORKSPACE = load("current-workspace.png")

NODE_NAMES = ["node-terminal.png", "node-evals.png", "node-designer.png", "node-training.png", "node-huggingface.png"]
NODES = [load(name) for name in NODE_NAMES]


def crop_resize(img: Image.Image, crop: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    return img.crop(crop).resize(size, Image.Resampling.LANCZOS)


CURRENT_FULL = CURRENT_WORKSPACE.resize((1600, int(1600 * CURRENT_WORKSPACE.height / CURRENT_WORKSPACE.width)), Image.Resampling.LANCZOS)
CURRENT_CANVAS = crop_resize(CURRENT_WORKSPACE, (245, 40, 1670, 1000), (1425, 960))
CURRENT_WORKFLOW = crop_resize(CURRENT_WORKSPACE, (330, 210, 1615, 820), (1285, 610))


@lru_cache(maxsize=160)
def peony_at(size: int) -> Image.Image:
    return PEONY.resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=80)
def node_at(index: int, size: int) -> Image.Image:
    return NODES[index].resize((size, size), Image.Resampling.LANCZOS)


def background(dark: bool = False) -> Image.Image:
    bg = Image.new("RGB", (W, H), INK if dark else CREAM)
    d = ImageDraw.Draw(bg)
    line = "#343138" if dark else "#ded9d2"
    for x in range(0, W, 48):
        d.line((x, 0, x, H), fill=line, width=1)
    for y in range(0, H, 48):
        d.line((0, y, W, y), fill=line, width=1)
    # Static grain dots are deterministic and make flat fields feel printed.
    for i in range(190):
        x = (i * 173 + 41) % W
        y = (i * 97 + 13) % H
        d.point((x, y), fill="#4b4750" if dark else "#c8c1ba")
    return bg


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str = LILAC_DARK,
          size: int = 24, anchor: str | None = None) -> None:
    draw.text(xy, text.upper(), font=font("mono", size), fill=fill, anchor=anchor,
              stroke_width=0)


def rule(draw: ImageDraw.ImageDraw, y: int, dark: bool = False) -> None:
    draw.line((80, y, W - 80, y), fill=LILAC_LIGHT if dark else INK, width=3)
    draw.rectangle((80, y - 7, 240, y + 7), fill=LILAC)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = PAPER,
         border: str = INK, shadow: str | None = INK, width: int = 4, offset: int = 10) -> None:
    x0, y0, x1, y1 = box
    if shadow:
        draw.rectangle((x0 + offset, y0 + offset, x1 + offset, y1 + offset), fill=shadow)
    draw.rectangle(box, fill=fill, outline=border, width=width)


def framed_image(frame: Image.Image, img: Image.Image, x: int, y: int, border: str = INK,
                 offset: int = 13, alpha: float = 1.0) -> None:
    d = ImageDraw.Draw(frame)
    w, h = img.size
    d.rectangle((x + offset, y + offset, x + w + offset, y + h + offset), fill=border)
    d.rectangle((x - 4, y - 4, x + w + 4, y + h + 4), fill=PAPER, outline=border, width=4)
    frame.paste(alpha_image(img, alpha), (x, y), alpha_image(img, alpha))


def headline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int,
             fill: str = INK, spacing: int = 0, anchor: str | None = None) -> None:
    draw.multiline_text(xy, text, font=font("serif-bold", size), fill=fill, spacing=spacing,
                        anchor=anchor)


def scene_hook(t: float) -> Image.Image:
    f = background(False)
    d = ImageDraw.Draw(f)
    p = ease(t / 0.8)
    # Botanical slab and offset registration marks.
    slab_x = int(1150 + (1 - p) * 420)
    d.rectangle((slab_x + 24, 96, 1860, 925), fill=INK)
    d.rectangle((slab_x, 72, 1836, 901), fill=LILAC_LIGHT, outline=INK, width=5)
    d.line((slab_x - 24, 520, 1885, 520), fill=PEACH, width=14)
    size = int(520 * pop(ease((t - 0.18) / 0.9)))
    pny = peony_at(max(2, size))
    px = slab_x + (686 - size) // 2
    py = 170 + (520 - size) // 2
    f.paste(pny, (px, py), pny)

    label(d, (110, 104), "BASHGYM / 01 / SOURCE MATERIAL", INK, 25)
    words = [
        ("YOUR BEST", 0.2, INK),
        ("CODING DATASET", 0.8, LILAC_DARK),
        ("ALREADY EXISTS.", 1.55, INK),
    ]
    y = 230
    for text, delay, color in words:
        a = ease((t - delay) / 0.45)
        x = int(110 - (1 - a) * 90)
        headline(d, (x, y), text, 76 if "DATASET" in text else 68, color)
        y += 118
    if t > 2.5:
        a = ease((t - 2.5) / 0.5)
        card(d, (110, 640, int(110 + 780 * a), 756), fill=INK, border=INK, shadow=LILAC, offset=10)
        if a > 0.75:
            d.text((145, 675), "IT'S THE WORK YOU DO EVERY DAY.", font=font("mono-bold", 30), fill=CREAM)
    rule(d, 952)
    return f


def draw_terminal_stream(f: Image.Image, t: float, x: int, y: int) -> None:
    d = ImageDraw.Draw(f)
    card(d, (x, y, x + 790, y + 700), fill=INK_2, border=LILAC_LIGHT, shadow="#09090b")
    d.rectangle((x, y, x + 790, y + 62), fill="#343139", outline=LILAC_LIGHT, width=3)
    for i, color in enumerate((PEACH, GOLD, MINT)):
        d.ellipse((x + 24 + 30 * i, y + 22, x + 38 + 30 * i, y + 36), fill=color)
    label(d, (x + 132, y + 18), "LIVE SESSION / TRACE CAPTURE", LILAC_LIGHT, 21)
    lines = [
        ("$ codex exec \"repair auth persistence\"", LILAC_LIGHT),
        ("→ inspect  bashgym/secrets.py", MINT),
        ("→ edit     settings_routes.py", PEACH),
        ("→ test     18 passed", "#89c997"),
        ("✓ outcome  verified", "#89c997"),
        ("→ decision preserve working token", LILAC),
        ("→ artifact trace_0710.json", MUTED),
    ]
    revealed = int(clamp(t / 4.2) * (len(lines) + 1))
    for i, (text, color) in enumerate(lines[:revealed]):
        yy = y + 102 + i * 70
        d.text((x + 34, yy), text, font=font("mono", 24), fill=color)
        if i == revealed - 1:
            d.rectangle((x + 34 + min(650, len(text) * 15), yy + 4, x + 50 + min(650, len(text) * 15), yy + 33), fill=LILAC)


def scene_capture(t: float) -> Image.Image:
    f = background(True)
    d = ImageDraw.Draw(f)
    if t < 7.2:
        label(d, (90, 74), "02 / CAPTURE THE WORK", LILAC_LIGHT, 25)
        headline(d, (90, 118), "EVERY SESSION\nBECOMES SIGNAL.", 60, CREAM, spacing=8)
        draw_terminal_stream(f, t, 90, 280)
        agents = ["CLAUDE CODE", "GEMINI CLI", "OPENCODE", "CODEX", "COPILOT CLI"]
        for i, name in enumerate(agents):
            a = ease((t - 0.55 - i * 0.45) / 0.45)
            x = int(1040 + (1 - a) * 260)
            y = 170 + i * 145
            card(d, (x, y, x + 690, y + 98), fill=CREAM, border=LILAC_LIGHT, shadow=LILAC_DARK, offset=8)
            d.text((x + 28, y + 27), name, font=font("mono-bold", 28), fill=INK)
            d.ellipse((x + 620, y + 34, x + 640, y + 54), fill="#70b985")
            # Moving context channel into the trace rail.
            d.line((x - 140, y + 49, x, y + 49), fill=LILAC, width=3)
            dot_x = int(x - 140 + ((t * 160 + i * 31) % 140))
            d.ellipse((dot_x - 6, y + 43, dot_x + 6, y + 55), fill=PEACH)
        label(d, (1385, 930), "LIVE + HISTORICAL IMPORT", LILAC_LIGHT, 22, anchor="mm")
    else:
        lt = t - 7.2
        label(d, (88, 70), "REAL WORK / ONE OPERATIONAL CANVAS", LILAC_LIGHT, 24)
        headline(d, (90, 116), "STRUCTURED TRACES.\nVISIBLE CONTEXT.", 58, CREAM)
        x = int(270 - (1 - ease(lt / 0.6)) * 320)
        framed_image(f, CURRENT_FULL, x, 230, border=LILAC_LIGHT)
        d.rectangle((1240, 890, 1805, 986), fill=LILAC, outline=CREAM, width=3)
        d.text((1522, 938), "TOOL CALLS · EDITS · COMMANDS · OUTCOMES", font=font("mono", 22), fill=INK, anchor="mm")
    return f


def scene_quality(t: float) -> Image.Image:
    if t < 6.8:
        f = background(False)
        d = ImageDraw.Draw(f)
        label(d, (90, 74), "03 / SEPARATE SIGNAL FROM NOISE", LILAC_DARK, 24)
        headline(d, (90, 118), "QUALITY IS A\nTRAINING SIGNAL.", 64, INK)
        metrics = ["SUCCESS", "VERIFICATION", "COMPLEXITY", "TOOL DIVERSITY", "EFFICIENCY", "LENGTH"]
        for i, name in enumerate(metrics):
            col, row = i % 2, i // 2
            x = 90 + col * 520
            y = 390 + row * 160
            a = ease((t - 0.3 - i * 0.2) / 0.5)
            card(d, (x, y, x + 470, y + 110), fill=PAPER, border=INK, shadow=LILAC_DARK, offset=int(10 * a))
            label(d, (x + 24, y + 18), name, LILAC_DARK, 20)
            d.rectangle((x + 24, y + 62, x + 438, y + 82), fill="#ddd7d0")
            d.rectangle((x + 24, y + 62, int(x + 24 + 380 * a), y + 82), fill=LILAC)
        # A current-canvas proof panel connects verification to the actual workspace.
        proof = CURRENT_WORKFLOW.resize((720, 342), Image.Resampling.LANCZOS)
        framed_image(f, proof, 1110, 360)
        d.rectangle((1170, 745, 1810, 850), fill=INK, outline=LILAC, width=4)
        d.text((1490, 797), "VERIFY BEFORE THE TRACE BECOMES TRAINING DATA", font=font("mono-bold", 20), fill=CREAM, anchor="mm")
        return f

    lt = t - 6.8
    f = background(True)
    d = ImageDraw.Draw(f)
    label(d, (82, 72), "VERIFIED TRACES → TRAINING DATA", LILAC_LIGHT, 24)
    headline(d, (84, 118), "CURATE. SEGMENT. DESIGN.", 58, CREAM)
    workflow = CURRENT_WORKFLOW.resize((1290, 612), Image.Resampling.LANCZOS)
    framed_image(f, workflow, 520, 250, border=LILAC_LIGHT)
    steps = ["VERIFY", "SCORE", "SEGMENT", "REDACT", "DESIGN"]
    for i, step in enumerate(steps):
        a = ease((lt - i * 0.25) / 0.45)
        x = int(75 - (1 - a) * 160)
        y = 300 + i * 132
        card(d, (x, y, x + 360, y + 88), fill=CREAM, border=LILAC_LIGHT, shadow=LILAC_DARK, offset=7)
        d.text((x + 102, y + 26), step, font=font("mono-bold", 25), fill=INK)
        idx = min(i, len(NODES) - 1)
        icon = node_at(idx, 58)
        f.paste(icon, (x + 20, y + 14), icon)
    d.rectangle((1320, 910, 1838, 996), fill=LILAC, outline=CREAM, width=3)
    d.text((1579, 953), "SFT DATA · DPO PAIRS · SYNTHETIC SEEDS", font=font("mono", 20), fill=INK, anchor="mm")
    return f


def scene_training(t: float) -> Image.Image:
    f = background(False)
    d = ImageDraw.Draw(f)
    label(d, (82, 70), "04 / TRAIN AN OPEN-WEIGHT SPECIALIST", LILAC_DARK, 24)
    headline(d, (84, 114), "ONE GYM.\nMULTIPLE LEARNING SIGNALS.", 60, INK)
    methods = ["SFT", "DPO", "GRPO", "RLVR", "DISTILL"]
    for i, method in enumerate(methods):
        a = ease((t - 0.25 - i * 0.3) / 0.45)
        x = int(80 - (1 - a) * 140)
        y = 375 + i * 112
        fill = LILAC if i in (0, 2, 3) else PAPER
        card(d, (x, y, x + 370, y + 78), fill=fill, border=INK, shadow=INK, offset=7)
        d.text((x + 28, y + 19), method, font=font("mono-bold", 28), fill=INK)

    # Current Training Run node expands into a model core and live metric rails.
    card(d, (610, 245, 1810, 870), fill=PAPER, border=INK, shadow=INK, offset=14)
    d.rectangle((610, 245, 1810, 340), fill="#efeae4", outline=INK, width=4)
    icon = node_at(3, 68)
    f.paste(icon, (642, 258), icon)
    d.text((735, 276), "TRAINING RUN", font=font("mono-bold", 30), fill=INK)
    d.text((1740, 280), "RUNNING", font=font("mono", 21), fill=LILAC_DARK, anchor="ra")
    pny = peony_at(250)
    f.paste(pny, (1080, 430), pny)
    d.ellipse((1055, 405, 1355, 705), outline=LILAC_DARK, width=7)
    d.ellipse((1010, 360, 1400, 750), outline=LILAC_LIGHT, width=3)
    rails = [
        ("TRAIN LOSS", 0.82, LILAC_DARK),
        ("PASS@1", 0.64, MINT),
        ("VERIFIED REWARD", 0.73, PEACH),
    ]
    for i, (name, value, color) in enumerate(rails):
        yy = 420 + i * 130
        label(d, (690, yy), name, INK, 19)
        d.rectangle((690, yy + 42, 1010, yy + 65), fill="#ded8d1")
        grow = ease((t - 1.2 - i * 0.3) / 1.2)
        d.rectangle((690, yy + 42, int(690 + 320 * value * grow), yy + 65), fill=color)
        d.text((990, yy), f"{value * grow:0.2f}", font=font("mono", 18), fill=INK, anchor="ra")
    for i, value in enumerate(("QLORA / 4-BIT", "CHECKPOINT 004", "GPU 72%")):
        yy = 420 + i * 130
        card(d, (1435, yy, 1740, yy + 78), fill=CREAM, border=INK, shadow=LILAC, offset=6)
        label(d, (1460, yy + 26), value, INK, 18)
    d.rectangle((1020, 930, 1818, 1010), fill=INK, outline=LILAC_LIGHT, width=3)
    options = ["LOCAL", "PRIVATE COMPUTE", "HUGGING FACE CLOUD"]
    for i, item in enumerate(options):
        x = 1060 + i * 245
        d.text((x, 956), item, font=font("mono", 18), fill=LILAC_LIGHT)
        if i < 2:
            d.text((x + 205, 956), "→", font=font("mono-bold", 22), fill=PEACH)
    return f


def scene_autoresearch(t: float) -> Image.Image:
    f = background(True)
    d = ImageDraw.Draw(f)
    label(d, (80, 70), "05 / AUTORESEARCH", LILAC_LIGHT, 24)
    headline(d, (82, 116), "IMPROVE THE RECIPE.\nNOT ONLY THE WEIGHTS.", 58, CREAM)
    x = int(560 + (1 - ease(t / 0.7)) * 300)
    card(d, (x, 220, 1815, 930), fill=INK_2, border=LILAC_LIGHT, shadow="#09090b", offset=12)
    d.rectangle((x, 220, 1815, 315), fill="#343139", outline=LILAC_LIGHT, width=4)
    d.text((x + 40, 250), "AUTORESEARCH / EXPERIMENT SEARCH", font=font("mono-bold", 25), fill=CREAM)
    d.rectangle((1630, 243, 1775, 292), fill=LILAC, outline=CREAM, width=2)
    d.text((1702, 268), "RUNNING", font=font("mono-bold", 18), fill=INK, anchor="mm")

    # Candidate recipe matrix: hyperparameters, trace mining, and schema search.
    headers = ["CANDIDATE", "LR", "RANK", "TRACE MIX", "SCHEMA", "SCORE"]
    col_x = [x + 34, x + 300, x + 440, x + 600, x + 840, x + 1060]
    for cx, text in zip(col_x, headers):
        label(d, (cx, 345), text, LILAC_LIGHT, 17)
    for row in range(6):
        yy = 395 + row * 76
        active = row == int((t * 1.25) % 6)
        d.rectangle((x + 25, yy, 1785, yy + 58), fill="#3c3844" if active else "#242329", outline=LILAC_DARK if active else "#4a4650", width=2)
        vals = [f"recipe_{row + 1:02d}", f"{(1 + row) * 2}e-5", str(8 + row * 4), f"gold {62 + row * 4}%", f"v{row + 3}", f"{0.61 + row * 0.047:0.3f}"]
        for cx, text in zip(col_x, vals):
            d.text((cx, yy + 18), text, font=font("mono", 17), fill=CREAM if active else "#a9a4b0")
    d.rectangle((x + 25, 875, 1785, 904), fill="#45414a")
    sweep = int((t * 260) % 1720)
    d.rectangle((x + 25, 875, x + 25 + sweep, 904), fill=LILAC)
    chips = ["HYPERPARAMETERS", "TRACE RECIPES", "DATA SCHEMAS"]
    for i, text in enumerate(chips):
        yy = 430 + i * 120
        card(d, (70, yy, 480, yy + 78), fill=CREAM, border=LILAC_LIGHT, shadow=LILAC_DARK, offset=7)
        label(d, (94, yy + 23), text, INK, 20)
    return f


def scene_export_route(t: float) -> Image.Image:
    f = background(False)
    d = ImageDraw.Draw(f)
    if t < 3.9:
        label(d, (86, 74), "06 / OWN THE ARTIFACTS", LILAC_DARK, 24)
        headline(d, (88, 120), "TRAINED HERE.\nYOURS EVERYWHERE.", 66, INK)
        exports = [("LORA", "ADAPTER"), ("MERGED", "WEIGHTS"), ("GGUF", "LOCAL SERVE")]
        for i, (big, small) in enumerate(exports):
            a = ease((t - 0.25 - i * 0.35) / 0.55)
            x = 120 + i * 570
            y = int(450 + (1 - a) * 220)
            card(d, (x, y, x + 470, y + 290), fill=LILAC if i == 2 else PAPER, border=INK, shadow=INK, offset=14)
            headline(d, (x + 38, y + 48), big, 56, INK)
            label(d, (x + 42, y + 150), small, INK, 21)
            d.line((x + 42, y + 210, x + 420, y + 210), fill=INK, width=3)
            d.text((x + 42, y + 226), "EXPORT READY  →", font=font("mono", 20), fill=INK)
    else:
        lt = t - 3.9
        label(d, (86, 74), "07 / CONFIDENCE-BASED ROUTING", LILAC_DARK, 24)
        headline(d, (88, 120), "THE RIGHT MODEL\nFOR THE RIGHT TASK.", 64, INK)
        y = 510
        card(d, (105, 390, 505, 690), fill=PAPER, border=INK, shadow=LILAC_DARK, offset=12)
        d.text((305, 455), "STUDENT", font=font("mono-bold", 32), fill=INK, anchor="mm")
        pny = peony_at(110)
        f.paste(pny, (250, 520), pny)
        card(d, (1415, 390, 1815, 690), fill=INK, border=INK, shadow=LILAC, offset=12)
        d.text((1615, 455), "TEACHER", font=font("mono-bold", 32), fill=CREAM, anchor="mm")
        d.text((1615, 555), "FALLBACK", font=font("serif-bold", 42), fill=LILAC_LIGHT, anchor="mm")
        d.line((520, y, 1400, y), fill=INK, width=8)
        confidence = 0.45 + 0.42 * (0.5 + 0.5 * math.sin(lt * 1.3))
        dotx = int(520 + 880 * confidence)
        d.ellipse((dotx - 20, y - 20, dotx + 20, y + 20), fill=LILAC, outline=INK, width=4)
        threshold_x = 520 + int(880 * 0.68)
        d.line((threshold_x, y - 75, threshold_x, y + 75), fill=PEACH, width=5)
        label(d, (threshold_x, y + 95), "CONFIDENCE THRESHOLD", INK, 18, anchor="mm")
        route_to_teacher = confidence < 0.68
        active_box = (1390, 365, 1840, 715) if route_to_teacher else (80, 365, 530, 715)
        d.rectangle(active_box, outline=LILAC_DARK, width=9)
        status = "FALL BACK" if route_to_teacher else "SERVE STUDENT"
        d.rectangle((690, 700, 1230, 800), fill=INK, outline=LILAC, width=4)
        d.text((960, 750), status, font=font("mono-bold", 29), fill=CREAM, anchor="mm")
        label(d, (960, 900), "PROGRESSIVE SHIFT · SAFE FALLBACK", LILAC_DARK, 24, anchor="mm")
    return f


def scene_flywheel(t: float) -> Image.Image:
    f = background(True)
    d = ImageDraw.Draw(f)
    label(d, (960, 80), "THE SELF-IMPROVEMENT LOOP", LILAC_LIGHT, 26, anchor="mm")
    headline(d, (960, 130), "CODE. VERIFY. CURATE. TRAIN. ROUTE. REPEAT.", 48, CREAM, anchor="ma")
    names = ["CODE", "VERIFY", "CURATE", "TRAIN", "ROUTE"]
    xs = [190, 555, 920, 1285, 1650]
    for i, (x, name) in enumerate(zip(xs, names)):
        a = ease((t - i * 0.18) / 0.48)
        size = int(160 * pop(a))
        icon = node_at(i, max(2, size))
        f.paste(icon, (x - size // 2, 400 - size // 2), icon)
        if i < 4:
            d.line((x + 105, 400, xs[i + 1] - 105, 400), fill=LILAC, width=7)
            pos = x + 105 + int((t * 220 + i * 70) % max(1, xs[i + 1] - x - 210))
            d.ellipse((pos - 9, 391, pos + 9, 409), fill=PEACH)
        d.text((x, 545), name, font=font("mono-bold", 29), fill=CREAM, anchor="mm")
        d.text((x, 590), f"0{i + 1}", font=font("mono", 21), fill=LILAC_LIGHT, anchor="mm")
    d.arc((430, 690, 1490, 1050), 190, 350, fill=LILAC, width=6)
    d.polygon([(1455, 850), (1510, 834), (1482, 888)], fill=PEACH)
    label(d, (960, 860), "EVERY NEW SESSION FEEDS THE NEXT MODEL", LILAC_LIGHT, 23, anchor="mm")
    return f


def scene_lockup(t: float) -> Image.Image:
    f = background(False if t < 2.4 else True)
    d = ImageDraw.Draw(f)
    dark = t >= 2.4
    primary = CREAM if dark else INK
    a = ease(t / 0.8)
    size = int(265 * pop(a))
    pny = peony_at(max(2, size))
    f.paste(pny, (235 - size // 2, 262 - size // 2), pny)
    headline(d, (420, 154), "BashGym", 118, primary)
    label(d, (430, 300), "SELF-IMPROVING AGENTIC DEVELOPMENT", LILAC_LIGHT if dark else LILAC_DARK, 23)
    rule(d, 380, dark=dark)
    if t > 0.7:
        headline(d, (110, 470), "TURN THE WAY YOU BUILD\nINTO THE SYSTEM THAT\nIMPROVES YOUR AI.", 64, primary, spacing=8)
    if t > 3.0:
        b = ease((t - 3.0) / 0.55)
        x1 = int(110 + 1700 * b)
        card(d, (110, 842, x1, 950), fill=LILAC, border=primary, shadow=PEACH, offset=12)
        if b > 0.82:
            d.text((960, 896), "CAPTURE THE WORK. OWN THE MODEL.", font=font("mono-bold", 31), fill=INK, anchor="mm")
    return f


SCENES = [
    (0.0, 5.0, scene_hook),
    (5.0, 18.0, scene_capture),
    (18.0, 32.0, scene_quality),
    (32.0, 43.0, scene_training),
    (43.0, 49.0, scene_autoresearch),
    (49.0, 57.5, scene_export_route),
    (57.5, 62.0, scene_flywheel),
    (62.0, 72.0, scene_lockup),
]


def render_frame(t: float) -> Image.Image:
    for start, end, renderer in SCENES:
        if start <= t < end:
            frame = renderer(t - start)
            break
    else:
        frame = scene_lockup(10.0)
    d = ImageDraw.Draw(frame)
    # Editorial top bar and scene-change registration sweep.
    d.rectangle((0, 0, W, 12), fill=LILAC)
    d.rectangle((0, H - 10, W, H), fill=PEACH)
    for boundary, _, _ in SCENES[1:]:
        delta = abs(t - boundary)
        if delta < 0.18:
            w = int(W * (1 - delta / 0.18))
            d.rectangle((0, 0, w, H), fill=LILAC)
            d.rectangle((max(0, w - 22), 0, w, H), fill=PEACH)
    return frame.convert("RGB")


def save_preview_frames() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for seconds in (1.5, 7.5, 13.0, 20.5, 28.0, 36.5, 45.5, 50.5, 55.0, 59.5, 65.0, 70.0):
        render_frame(seconds).save(FRAMES_DIR / f"frame-{seconds:04.1f}.jpg", quality=92)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_preview_frames()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    total = int(DURATION * FPS)
    try:
        for i in range(total):
            frame = render_frame(i / FPS)
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            if i % (FPS * 5) == 0:
                print(f"render {i / FPS:05.1f}s / {DURATION:.1f}s", flush=True)
    finally:
        proc.stdin.close()
    code = proc.wait()
    if code:
        raise SystemExit(code)
    print(OUT)


if __name__ == "__main__":
    main()
