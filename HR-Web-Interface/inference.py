"""
Illumination-routed HR inference: loads both PhysNet checkpoints once at
startup, then for each uploaded video picks the checkpoint matching the
measured face-region illumination and runs the full pipeline end to end.
"""
import os
import numpy as np
import torch

from physnet_model import PhysNet_padding_Encoder_Decoder_MAX
from preprocess import preprocess_video, CHUNK_LENGTH
from postprocess import rppg_to_bpm, aggregate_bpm

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
NORMAL_LIGHT_CKPT = os.path.join(MODELS_DIR, "UBFC_UBFC_physnet_Epoch25.pth")
LOW_LIGHT_CKPT = os.path.join(MODELS_DIR, "MMPD_5FOLD_physnet_fold3_Epoch6.pth")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models = {"normal": None, "low": None}


def _load_model(checkpoint_path):
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=CHUNK_LENGTH)
    state_dict = torch.load(checkpoint_path, map_location=_device)
    # rPPG-Toolbox saves plain state_dicts (sometimes wrapped in DataParallel
    # with a 'module.' prefix) -- handle both.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()
    return model


def load_models():
    """Load both checkpoints into memory. Call once at Flask startup."""
    missing = [p for p in (NORMAL_LIGHT_CKPT, LOW_LIGHT_CKPT) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint file(s): " + ", ".join(missing) +
            ". Place both .pth files in the models/ folder before starting the app."
        )
    _models["normal"] = _load_model(NORMAL_LIGHT_CKPT)
    _models["low"] = _load_model(LOW_LIGHT_CKPT)


def _run_physnet(model, chunk):
    """chunk: np.ndarray [T, H, W, 3] float32 (diff-normalized) -> rPPG [T] np.ndarray."""
    # PhysNet expects [B, C, T, H, W]
    tensor = torch.from_numpy(chunk).permute(3, 0, 1, 2).unsqueeze(0).float().to(_device)
    with torch.no_grad():
        rppg, _, _, _ = model(tensor)
    return rppg.squeeze(0).cpu().numpy()


def estimate_heart_rate(video_path):
    """Full pipeline for one uploaded video. Returns a result dict with the
    final BPM, the routing decision, and per-chunk diagnostics.

    PhysNet forward passes happen per 128-frame chunk (fixed by the
    architecture -- FRAME_NUM=128). Two ways to turn those per-chunk rPPG
    outputs into one BPM were tried empirically:
      1. Concatenate all chunks into one continuous signal, then a single
         FFT (this is what rPPG-Toolbox's own evaluation/metrics.py does).
         Gives fine frequency resolution "for free", but stitching together
         independent forward passes can introduce small discontinuities at
         chunk boundaries that inject spurious frequency content.
      2. Compute BPM per chunk independently (avoids boundary artifacts),
         but zero-pad each chunk's FFT well beyond 128 samples so the peak
         isn't forced onto one of only ~7 coarse ~14 BPM-wide bins, then
         take the median across chunks.

    Tested against 8 real ground-truth-labeled clips (4 UBFC, 4 MMPD
    low-light), the two checkpoints disagreed on which method is more
    accurate: full-clip concatenation (1) averaged 2.8 BPM error on UBFC vs
    4.8 for per-chunk median; per-chunk median (2) averaged 5.5 BPM error on
    MMPD vs 8.2 for full-clip. So the primary "bpm" is chosen per route
    based on that evidence. Both values are still returned so this can be
    re-evaluated if more labeled test data becomes available -- 4 clips per
    checkpoint is a small sample, not a proof.
    """
    if _models["normal"] is None or _models["low"] is None:
        load_models()

    prep = preprocess_video(video_path)
    chunks = prep["chunks"]
    if len(chunks) == 0:
        raise ValueError("Video did not yield any full 128-frame chunks after preprocessing.")

    route = "low" if prep["is_low_light"] else "normal"
    model = _models[route]

    rppg_segments = []
    per_chunk_bpm = []
    for chunk in chunks:
        rppg = _run_physnet(model, chunk)
        rppg_segments.append(rppg)
        per_chunk_bpm.append(rppg_to_bpm(rppg, fs=30))

    median_bpm = aggregate_bpm(per_chunk_bpm)

    full_rppg = np.concatenate(rppg_segments)
    full_clip_bpm = rppg_to_bpm(full_rppg, fs=30)

    # Empirically route-dependent: see docstring above for the numbers.
    final_bpm = full_clip_bpm if route == "normal" else median_bpm

    return {
        "bpm": round(final_bpm, 1),
        "bpm_full_clip": round(full_clip_bpm, 1),
        "per_chunk_bpm": [round(b, 1) for b in per_chunk_bpm],
        "route": "low_light (MMPD)" if route == "low" else "normal_light (UBFC)",
        "illumination": round(prep["illumination"], 1),
        "face_found_ratio": round(prep["face_found_ratio"], 2),
        "n_frames": prep["n_frames"],
        "n_chunks": len(chunks),
    }
