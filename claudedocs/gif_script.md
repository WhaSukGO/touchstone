# GIF / screencast script

Goal: a ~20–30s loopable terminal GIF that lands the hook — **"watch it catch an agent
cheating."** The cheat demo is offline + instant, so the GIF needs no GPU, API key, or
editing tricks. Best tool: **VHS** (charmbracelet/vhs) — you write a `.tape` and it renders
a deterministic GIF. An asciinema route is below for those who prefer it.

---

## Primary GIF — "catch the cheat" (VHS)

Install VHS: `brew install vhs` (or see the repo). Then `vhs cheat.tape`. Save this as
`cheat.tape`:

```tape
# cheat.tape  →  vhs cheat.tape  →  cheat.gif
Output cheat.gif
Set FontSize 17
Set Width 1180
Set Height 680
Set Padding 20
Set Theme "Dracula"
Set TypingSpeed 55ms

# a clean prompt helps; assumes you're in the repo with deps installed
Hide
Type "clear" Enter
Show

Type "# AI agents cheat their evals. Touchstone catches them. (no GPU, no API key)"
Enter
Sleep 1.2s
Type "python -m lab.run_cheat_demo"
Sleep 600ms
Enter
Sleep 3.5s          # table renders (offline run is ~1–2s; padding for safety)
Sleep 4s            # HOLD on the result — the two REJECTED ← fools naive lines
Sleep 1s
```

What the viewer sees: the command, then the table where **hardcode** and **special-case**
score `100%` on the visible tests and `8%` on the hidden ones → **REJECTED**, with the
honest one VERIFIED. That contrast is the whole pitch.

> Tip: the punchline is the `← fools naive` rows. Keep the final HOLD long enough to read
> them (≥4s). If the GIF feels slow, drop `TypingSpeed` to `35ms`.

---

## Secondary GIF — "the autograder" (VHS)

For a second asset (the "useful tool" angle): `python -m lab.run_autograde`. Save as
`autograde.tape`, same header, body:

```tape
Type "# Grade an AI coding agent — on tests it can't see or game."
Enter
Sleep 1.2s
Type "python -m lab.run_autograde"
Sleep 600ms
Enter
Sleep 4s
Sleep 4s            # HOLD: "claimed 3/3 … verified 1/3 on hidden tests"
```

Money line to land: *"the submitter would have CLAIMED 3/3 … Touchstone verified 1/3 on
hidden tests."*

---

## Asciinema fallback (manual)

```bash
# 1) record (clean, ~80x24 terminal, big readable font)
asciinema rec touchstone.cast
#    then, in the recording shell:
clear
python -m lab.run_cheat_demo
#    wait ~4s after it finishes so the result is readable, then:
exit

# 2) turn the cast into a GIF
agg --font-size 28 --theme dracula touchstone.cast touchstone.gif
#    (agg = asciinema's gif generator: cargo install --git https://github.com/asciinema/agg)
```

---

## Recording tips (any tool)

- **Clean shell**: set a minimal prompt so the repo path/venv doesn't clutter the frame —
  e.g. `PS1='$ '` for the recording session.
- **Size**: ~1100–1200 px wide, font 17–28 so it's legible inline on GitHub/X.
- **Keep it short**: ≤30s, and end on the result (the table), not a fresh prompt — GIFs loop.
- **No secrets on screen**: the cheat/autograde demos need no API key, so nothing sensitive
  is shown. (For a `--live` recording, set the key via `.env`/`set -a` *before* recording,
  off-camera.)
- **One idea per GIF**: cheat-demo for the hook; autograder for "it's a real tool." Don't
  cram both into one.

## Where to use them

- Top of the README and DEMO.md (inline `![demo](cheat.gif)`).
- The Show HN / blog post (writeup_draft.md) — the cheat GIF right under the title.
- X/LinkedIn — the cheat GIF as the lead media of the thread.

## Suggested capture order (a 3-beat screen recording, if you want a longer video)
1. `python -m lab.run_cheat_demo` — the hook (cheats caught).
2. `python -m lab.run_autograde` — the tool (claimed 3/3 → verified 1/3).
3. `python -m lab.run_humaneval --live` — the credibility (a real agent on HumanEval+).
   (This one is billed + slow; pre-run it and show the captured output, or speed it up.)
