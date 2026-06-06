# genart-render

Self-contained **generative-art render engines** (procedural visuals + original/PD music), run on
GitHub Actions for free, unlimited compute (public repo). All code is original numpy implementations;
soundfonts are free/PD (GeneralUser GS; YDP Grand Piano, CC-BY FreePats) fetched at render time.

Trigger a render from the **Actions** tab → "render" → Run workflow → paste a command, e.g.:

```
compose.py --style lofi --wav outputs/lofi.wav --roll outputs/lofi.mp4 --bars 24
physics_sim.py --scene hexagon --output outputs/hex.mp4 --seconds 60
midi_render.py --midi <url-or-path>.mid --wav outputs/x.wav --roll outputs/x.mp4 --gm
```

The output mp4/wav is uploaded as a downloadable **artifact**. No business data here — just the engines.
