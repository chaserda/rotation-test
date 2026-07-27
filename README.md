# Rotation Counter

Count full **360° human rotations** in a video using a vision LLM for per-frame orientation labels and deterministic Python for temporal counting.

OpenCV is used only to sample/resize frames. No YOLO, MediaPipe, or optical flow.

## Approach

1. Sample frames from the video (OpenCV preprocess only)
2. VLM labels each frame: `front` / `back` / `side` (face-first)
3. Python counts complete laps: **leave front → see back → return to front**

Only a `front` label closes a lap. Partial endings (e.g. 1.5 ending on back) stay incomplete as full rotations, but the final summary also reports skateboard-style degrees (e.g. **540°**).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add API keys to .env
```

## Available videos 
| Video | Expected count |
|---|---|
| `videos/1.5rotationsTest.mp4` | 1 (540°) |
| `videos/3rotationsTest.mp4` | 3 (1080°) |
| `videos/5rotationsTest.mp4` | 5 (1800°) |

## Choose a provider 

When running the script you can append a --provider argument with the provider you would like, for example:
```python main.py videos/3rotationsTest.mp4 --provider openai``` or ```python main.py videos/3rotationsTest.mp4 --provider openai```
in order to use different providers, you will need to at your API keys to the .env file.


## Run

```bash
python main.py videos/5rotationsTest.mp4
python main.py videos/3rotationsTest.mp4 --provider openai
python main.py videos/1.5rotationsTest.mp4 --provider claude
```

Expected on provided clips:

| Video | Expected count |
|---|---|
| `videos/1.5rotationsTest.mp4` | 1 (540°) |
| `videos/3rotationsTest.mp4` | 3 (1080°) |
| `videos/5rotationsTest.mp4` | 5 (1800°) |

## Providers

| `--provider` | Env key | Default model |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` | `gemini-3.5-flash-lite` |
| `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini` |
| `claude` | `CLAUDE_API_KEY` | `claude-sonnet-5` |

Optional: `VLM_PROVIDER`, `CLASSIFY_WORKERS` (default `8`). See `.env.example`.

## Layout

```
main.py                         CLI
rotation_counter/
  extract.py                    frame sampling
  vlm.py                        prompts + parallel classify
  count.py                      deterministic counter + degree measure
  providers/                    gemini / openai / claude
tests/test_count.py             unit tests (no API calls)
notes.txt                       design decisions / interview notes
```

## Tests

```bash
python -m unittest tests.test_count -v
```

## Design notes

- **LLM never counts** — it only labels frames; Python owns cycle math
- **Classify mode** — one face-first call per frame, in parallel
- **Open-lap finalize** — if a lap is open and the clip does not end on `back`, re-check ending frames alone (does not force `front`)
- **Degrees** — printed once at the end: `full * 360` plus open-lap partial (180/270)
