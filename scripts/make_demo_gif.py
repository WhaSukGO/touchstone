"""Render an animated GIF of a Touchstone demo (no external recording tools needed).

Runs the demo, then draws its output onto terminal-style frames (progressive reveal + a
long hold) with Pillow.

  python scripts/make_demo_gif.py            # cheat demo  -> assets/cheat.gif
  python scripts/make_demo_gif.py reward     # verifier-problem demo -> assets/reward_hacking.gif
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

TARGETS = {
    "cheat": ("lab.run_cheat_demo", "assets/cheat.gif"),
    "reward": ("lab.run_reward_hacking_demo", "assets/reward_hacking.gif"),
}

BG = (30, 30, 46)
DIM = (108, 112, 134)
TEXT = (205, 214, 244)
TITLE = (249, 226, 175)
GREEN = (166, 227, 161)
RED = (243, 139, 168)
PROMPT = (137, 220, 235)

FS = 20
PAD = 26


def color_for(line: str):
    s = line.strip()
    if "REJECTED" in line:
        return RED
    if "VERIFIED" in line:
        return GREEN
    if any(k in line for k in ("VERIFIER PROBLEM", "REWARD HACKING", "TOUCHSTONE vs")):
        return TITLE
    if set(s) <= set("─ ") or s.startswith(
            ("reward", "naive", "RLVR", "Each", "with", "solution", "task:", "mode:",
             "(visible)", "(hidden)", "every", "result")):
        return DIM
    return TEXT


def main(argv: list[str]) -> int:
    key = argv[0] if argv else "cheat"
    module, out_rel = TARGETS[key]
    out = ROOT / out_rel

    body = subprocess.run([sys.executable, "-m", module], cwd=ROOT,
                          capture_output=True, text=True).stdout.splitlines()
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    lines = [f"$ python -m {module}", ""] + body
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

    counts, durations = [1], [900]
    n = 2
    while n < len(lines):
        n = min(len(lines), n + 2)
        counts.append(n)
        durations.append(170)
    counts.append(len(lines))
    durations.append(5000)

    frames = [frame(c) for c in counts]
    base = frames[-1].quantize(colors=64)
    pframes = [f.quantize(palette=base) for f in frames]

    out.parent.mkdir(parents=True, exist_ok=True)
    pframes[0].save(out, save_all=True, append_images=pframes[1:], duration=durations,
                    loop=0, disposal=2, optimize=True)
    print(f"wrote {out} ({width}x{height}, {len(frames)} frames, {out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
