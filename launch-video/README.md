# BashGym launch video

This folder is an editable, local production package for the BashGym product-launch film.

## Build

```powershell
.\.venv\Scripts\python.exe build.py
```

The build renders the picture master, generates an original music stem, creates or reuses the neural voiceover, mixes the final exports, and writes verification metadata to `output/`.

## Swapping music

Use `output/bashgym-launch-vo-only.mp4` as the picture-and-voice base, or replace `output/music-original.wav` and rerun the final mix command printed by `build.py`. The voiceover is supplied separately as both MP3 and 48 kHz stereo WAV.

## Source assets

The product frame is the current connected-workspace screenshot supplied for this production. Logo and node art are copied from `frontend/public/`. Older repository screenshots are intentionally not used in the render.
