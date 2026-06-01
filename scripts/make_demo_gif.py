"""Render an animated GIF of the cheat demo (no external recording tools needed).

Runs `python -m lab.run_cheat_demo`, then draws its output onto terminal-style frames
(progressive reveal + a long hold on the result) with Pillow. Output: assets/cheat.gif.

  python scripts/make_demo_gif.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
OUT = ROOT / "assets" / "cheat.gif"

BG = (30, 30, 46)        # base
DIM = (108, 112, 134)    # separators / sub-headers
TEXT = (205, 214, 244)   # default
TITLE = (249, 226, 175)  # yellow
GREEN = (166, 227, 161)  # VERIFIED
RED = (243, 139, 168)    # REJECTED
PROMPT = (137, 220, 235) # the typed command

FS = 20
PAD = 26


def color_for(line: str):
    s = line.strip()
    if "REJECTED" in line:
        return RED
    if "VERIFIED" in line:
        return GREEN
    if s.startswith("TOUCHSTONE"):
        return TITLE
    if set(s) <= set("─ ") or s.startswith(("task:", "solution", "(visible)")):
        return DIM
    return TEXT


def main() -> int:
    out = subprocess.run([sys.executable, "-m", "lab.run_cheat_demo"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    body = [ln for ln in out.splitlines()]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    lines = ["$ python -m lab.run_cheat_demo", ""] + body
    font = ImageFont.truetype(FONT_PATH, FS)
    cw = font.getlength("M")
    lh = FS + 8
    width = int(max(len(ln) for ln in lines) * cw) + 2 * PAD
    height = lh * len(lines) + 2 * PAD

    def frame(n_visible: int) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines[:n_visible]):
            col = PROMPT if i == 0 else color_for(ln)
            d.text((PAD, PAD + i * lh), ln, font=font, fill=col)
        return img

    # frames: prompt, then reveal the output ~2 lines at a time, then a long hold
    counts, durations = [], []
    counts.append(1); durations.append(900)              # the command
    n = 2
    while n < len(lines):
        n = min(len(lines), n + 2)
        counts.append(n); durations.append(170)
    counts.append(len(lines)); durations.append(5000)    # hold on the result

    frames = [frame(c) for c in counts]
    base = frames[-1].quantize(colors=64)
    pframes = [f.quantize(palette=base) for f in frames]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pframes[0].save(OUT, save_all=True, append_images=pframes[1:], duration=durations,
                    loop=0, disposal=2, optimize=True)
    print(f"wrote {OUT} ({width}x{height}, {len(frames)} frames, {OUT.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
