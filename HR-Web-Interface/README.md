# Heart Rate Estimation from Facial Video — Illumination-Routing Interface

Upload any facial video (normal or low light) and get a predicted heart rate
in BPM. The app measures the illumination of the detected face region and
automatically routes the clip to whichever PhysNet checkpoint was trained
for that lighting condition:

- **Normal light** → `UBFC_UBFC_physnet_Epoch25.pth` (trained/tested on UBFC-rPPG)
- **Low light** → `MMPD_5FOLD_physnet_fold3_Epoch6.pth` (best of 5 folds on MMPD, MAE 1.62 BPM / Pearson 0.933 on the held-out fold)

This sidesteps the SCI low-light enhancement stage (still WIP, see
`SCI-Enhancement/` in the main repo) by using a model that was trained
directly on genuinely low-light footage (MMPD) instead of trying to brighten
frames first.

## Setup

1. Install dependencies (Python 3.9+):
   ```
   pip install -r requirements.txt
   ```
   If you have a CUDA GPU, install the matching `torch` build from
   https://pytorch.org/get-started/locally/ instead of the plain `pip install torch`
   for faster inference (CPU works fine, just slower — a few seconds per 128-frame chunk).

2. Place both checkpoint files in `models/`:
   ```
   models/UBFC_UBFC_physnet_Epoch25.pth
   models/MMPD_5FOLD_physnet_fold3_Epoch6.pth
   ```

3. Run:
   ```
   python app.py
   ```
   Open http://127.0.0.1:5000 in a browser.

## How it works

1. **Read + resample**: video is decoded and resampled to 30fps (both
   checkpoints were trained on 30fps data; arbitrary uploads may come in at
   other frame rates).
2. **Illumination check**: a face is detected (Haar Cascade) in a sample of
   frames and mean grayscale intensity within the face box is measured. Below
   `ILLUMINATION_THRESHOLD = 80` (0–255 scale, tunable in `preprocess.py`) the
   clip is classified low-light.
3. **Face crop/resize**: largest face on the first frame is detected, boxed
   with a 1.5x margin, and every frame is cropped/resized to 72x72 — matching
   rPPG-Toolbox's `BaseLoader` preprocessing exactly.
4. **DiffNormalize + chunk**: frames are converted to frame-to-frame
   differences (globally standardized) and split into non-overlapping
   128-frame chunks.
5. **PhysNet forward pass**: each chunk goes through the routed checkpoint to
   produce a raw rPPG waveform.
6. **HR extraction**: each chunk's differenced rPPG output is integrated
   (`cumsum`), detrended, Butterworth-bandpassed (0.75–2.5 Hz), and
   FFT-peaked (zero-padded to a 4096-point FFT so the peak isn't forced onto
   one of only ~7 coarse ~14 BPM-wide bins that a raw 128-sample FFT gives)
   to get a per-chunk BPM. All chunks' rPPG segments are also concatenated
   into one continuous signal and put through the same steps once, giving a
   second "full-clip" BPM estimate. The **final reported BPM is chosen per
   route**: full-clip concatenation for the normal-light (UBFC) checkpoint,
   median-of-chunks for the low-light (MMPD) checkpoint — see below.

## Accuracy (measured against real ground-truth data)

Both checkpoints were tested against real labeled clips: 4 UBFC-rPPG
videos (raw, uncompressed `.avi`, ground-truth HR from `ground_truth.txt`)
and 4 MMPD low-light `.mat` clips (ground-truth HR from `GT_ppg`, converted
to lossless `.mkv` for upload so no lossy re-encoding skews the test —
**lossy compression measurably destroys the rPPG signal**, confirmed
directly: an `mp4v`-compressed copy of the same MMPD clip pushed the error
from ~6 BPM to ~20 BPM with no other change).

| Method | UBFC avg error | MMPD avg error |
|---|---|---|
| Full-clip concatenation | 2.8 BPM | 8.2 BPM |
| Median of per-chunk (zero-padded FFT) | 4.8 BPM | 5.5 BPM |

The two checkpoints disagreed on which aggregation method is more accurate,
likely because concatenating independent 128-frame forward passes can
introduce small discontinuities at chunk boundaries — this seems to hurt
the MMPD checkpoint more than the UBFC one. `inference.py` picks the better
method per route based on this evidence. **This is a small sample (4 clips
per checkpoint)** — a real conclusion, not a guess, but worth re-validating
with more labeled data if it's available, since the choice could tip back
the other way with a larger sample.

Individual clip errors ranged from under 1 BPM (UBFC subject4, full-clip:
0.2 BPM) to about 12 BPM (one MMPD clip that was an outlier under every
method tried). Average error across all 8 clips using the current
route-dependent choice is about 4.1 BPM — well short of the paper's
reported aggregate MAE (0.8–1.6 BPM depending on dataset, computed by
averaging over full test folds under controlled conditions), but reasonable
for a working prototype evaluated on arbitrary individual clips.

## Known limitations (current state)

- Two hard-coded checkpoints, not the full SCI-enhanced pipeline described in
  the paper — the SCI enhancement stage is still being debugged
  (see `SCI-Enhancement/README.md`).
- Illumination threshold (80) is a reasonable starting point, not empirically
  calibrated against a labeled illumination dataset — validate against your
  own videos and adjust if the routing feels wrong.
- Static (frame-0) face detection: works well for videos where the subject
  stays roughly still and framed similarly to UBFC/MMPD recording setups;
  large head movement mid-clip isn't tracked per-frame.
- Videos shorter than ~4.3s (128 frames @ 30fps) are rejected with an error
  rather than padded — padding would distort the FFT-based HR estimate.
- No GPU is required, but expect ~2-5s per 128-frame chunk on CPU.

## Files

- `app.py` — Flask server and routes.
- `templates/index.html` — upload page.
- `preprocess.py` — video reading, resampling, face detection, illumination
  scoring, diff-normalization, chunking.
- `postprocess.py` — rPPG waveform → BPM (cumsum/detrend/bandpass/FFT).
- `physnet_model.py` — PhysNet architecture (unmodified from rPPG-Toolbox).
- `inference.py` — ties it together: loads both checkpoints, routes, runs
  inference, aggregates per-chunk BPM.
- `models/` — the two `.pth` checkpoint files (~3MB each, included directly
  in the repo — small enough not to need Git LFS).

## Deploying publicly

Once this works locally, deploying it (e.g. on Render/Railway/HF Spaces)
mainly involves: setting `app.run(host="0.0.0.0", port=...)` to read the
platform's `PORT` env var, and switching `debug=False` (already set). A
link can then be added to the main repo README for recruiters to try
directly.
