"""
Preprocessing pipeline for the illumination-routing HR interface.

Faithfully follows rPPG-Toolbox's BaseLoader conventions (Haar Cascade face
detection, 72x72 crop/resize, DiffNormalized frames, 128-frame chunking) so
that inference matches the preprocessing the checkpoints were trained on.
"""
import os
import cv2
import numpy as np

# Haar Cascade classifier used by rPPG-Toolbox by default (BACKEND: 'HC')
_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

FACE_SIZE = 72
CHUNK_LENGTH = 128
LARGER_BOX_COEF = 1.5

# Mean pixel intensity threshold (0-255, grayscale) in the detected face
# region used to route between the normal-light (UBFC) and low-light (MMPD)
# checkpoints. Tunable; see README for how this was picked.
ILLUMINATION_THRESHOLD = 80.0


# Hard cap on output frames (after 30fps resampling), regardless of the
# uploaded file's actual length. Kept low on Render (30s) to bound memory on
# its 512MB free tier; restored to the full 60s used during accuracy
# testing when running locally, where there's no such constraint. Override
# with the MAX_SOURCE_FRAMES_OVERRIDE env var if needed.
_ON_RENDER = os.environ.get("RENDER") == "true"
_DEFAULT_MAX_SOURCE_FRAMES = 900 if _ON_RENDER else 1800
MAX_SOURCE_FRAMES = int(os.environ.get("MAX_SOURCE_FRAMES_OVERRIDE", _DEFAULT_MAX_SOURCE_FRAMES))


def _compute_target_indices(total_frames, src_fps, max_output_frames):
    """Precompute which source-frame indices to keep, resampling to 30fps
    and capping output length, WITHOUT needing to load any frames first.
    Returns None if total_frames is unknown/unreliable (some containers
    misreport it) -- caller falls back to a straight sequential cap with no
    fps correction rather than crashing.
    """
    target_fps = 30.0
    if total_frames is None or total_frames <= 0:
        return None
    if src_fps is None or src_fps <= 0:
        src_fps = target_fps
    duration_sec = total_frames / src_fps
    n_target = int(round(duration_sec * target_fps))
    n_target = max(1, min(n_target, max_output_frames))
    indices = np.linspace(0, total_frames - 1, num=n_target)
    indices = np.clip(np.round(indices).astype(int), 0, total_frames - 1)
    return indices.tolist()


def detect_largest_face(frame_color):
    """Return (x, y, w, h) of the largest detected face, or None.

    Matches BaseLoader.face_detection() exactly: detectMultiScale is called
    with OpenCV's default params (scaleFactor=1.1, minNeighbors=3, no
    minSize) on the color frame directly (not pre-converted to grayscale --
    OpenCV converts internally), and ties are broken by box width rather
    than area (equivalent for the near-square boxes Haar cascade returns).
    Using different detectMultiScale params than training produces a
    different face box and measurably hurts accuracy, so this must stay in
    sync with BaseLoader.py if that ever changes.
    """
    faces = _face_cascade.detectMultiScale(frame_color[:, :, :3].astype(np.uint8))
    if len(faces) == 0:
        return None
    return faces[int(np.argmax(faces[:, 2]))]


def enlarge_box(box, frame_shape, coef=LARGER_BOX_COEF):
    """Expand a face box by `coef` around its center, clipped to frame bounds."""
    x, y, w, h = box
    H, W = frame_shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w, new_h = w * coef, h * coef
    new_x = max(0, int(cx - new_w / 2.0))
    new_y = max(0, int(cy - new_h / 2.0))
    new_x2 = min(W, int(cx + new_w / 2.0))
    new_y2 = min(H, int(cy + new_h / 2.0))
    return new_x, new_y, new_x2 - new_x, new_y2 - new_y


def read_crop_and_measure(video_path, illum_sample_stride=5, use_larger_box=True):
    """Single streaming pass over the video: reads each frame, resamples to
    30fps and caps length, crops/resizes to FACE_SIZE x FACE_SIZE, and
    samples illumination -- all without ever holding the full-resolution
    decoded video in memory at once.

    This matters a lot on memory-constrained deployments (e.g. Render's free
    512MB RAM tier): a few seconds of 720p+ video decoded to raw uint8
    frames can easily be 300-400MB by itself, on top of PyTorch/OpenCV's own
    baseline memory use, before any processing even starts. The 72x72
    cropped representation the model actually needs is tiny by comparison,
    so we crop immediately per-frame and discard the full-resolution frame
    right away instead of accumulating a big array first and cropping after.

    Face detection runs once, on the first kept frame (matches BaseLoader's
    default static/non-dynamic detection), and that box is reused for every
    subsequent frame. Illumination is sampled independently (its own
    per-sample face detection, for a more representative + diagnostic
    face_found_ratio) every `illum_sample_stride`-th kept frame.

    Returns dict with: cropped [N,72,72,3] float32, fps=30.0, n_frames,
    illumination, face_found_ratio.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_indices = _compute_target_indices(total_frames_hint, src_fps, MAX_SOURCE_FRAMES)

    box = None
    cropped_frames = []
    illum_face_vals = []
    illum_whole_vals = []
    faces_found = 0
    illum_checked = 0

    src_idx = 0
    target_ptr = 0
    kept_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if target_indices is not None:
            keep = target_ptr < len(target_indices) and src_idx == target_indices[target_ptr]
            if keep:
                target_ptr += 1
        else:
            # Unknown/unreliable video length metadata: fall back to a
            # straight sequential cap with no fps resampling, rather than
            # crashing. Rare in practice for normal uploads.
            keep = kept_count < MAX_SOURCE_FRAMES

        if keep:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if box is None:
                detected = detect_largest_face(frame)
                if detected is None:
                    H, W = frame.shape[:2]
                    side = min(H, W)
                    y0, x0 = (H - side) // 2, (W - side) // 2
                    detected = (x0, y0, side, side)
                box = enlarge_box(detected, frame.shape) if use_larger_box else detected

            x, y, w, h = box
            w, h = max(1, w), max(1, h)
            crop = frame[y:y + h, x:x + w, :]
            if crop.size == 0:
                crop = frame
            resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
            cropped_frames.append(resized.astype(np.float32))

            if kept_count % illum_sample_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                illum_whole_vals.append(gray.mean())
                illum_checked += 1
                ibox = detect_largest_face(frame)
                if ibox is not None:
                    ix, iy, iw, ih = enlarge_box(ibox, frame.shape)
                    face_crop = gray[iy:iy + ih, ix:ix + iw]
                    if face_crop.size > 0:
                        illum_face_vals.append(face_crop.mean())
                        faces_found += 1

            kept_count += 1
            if target_indices is not None and target_ptr >= len(target_indices):
                break
            if target_indices is None and kept_count >= MAX_SOURCE_FRAMES:
                break

        src_idx += 1

    cap.release()

    if len(cropped_frames) == 0:
        raise ValueError("No frames could be read from the video.")

    face_found_ratio = faces_found / max(1, illum_checked)
    illumination = float(np.mean(illum_face_vals)) if illum_face_vals else float(np.mean(illum_whole_vals))

    return {
        "cropped": np.asarray(cropped_frames, dtype=np.float32),
        "n_frames": kept_count,
        "illumination": illumination,
        "face_found_ratio": face_found_ratio,
    }


def diff_normalize_data(data):
    """Exact port of BaseLoader.diff_normalize_data: frame-to-frame
    differences normalized by (sum + eps), then globally standardized,
    with the last frame zero-padded.
    """
    n, h, w, c = data.shape
    normalized_len = n - 1
    normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
    for j in range(normalized_len):
        denom = (data[j + 1] + data[j] + 1e-7)
        normalized_data[j] = (data[j + 1] - data[j]) / denom
    normalized_data = normalized_data / np.std(normalized_data)
    normalized_data[np.isnan(normalized_data)] = 0
    normalized_data = np.append(normalized_data, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
    return normalized_data


def chunk_frames(data, chunk_length=CHUNK_LENGTH):
    """Split into non-overlapping chunks of `chunk_length` frames, dropping
    any remainder (matching BaseLoader.chunk())."""
    n = data.shape[0]
    n_chunks = n // chunk_length
    if n_chunks == 0:
        return []
    return [data[i * chunk_length:(i + 1) * chunk_length] for i in range(n_chunks)]


def preprocess_video(video_path):
    """Full pipeline: streaming read + resample to 30fps + crop/resize +
    illumination check, then diff-normalize -> chunk into model-ready
    tensors. See read_crop_and_measure() for why this is one streaming pass
    instead of separate read/resample/crop steps (memory).
    """
    result = read_crop_and_measure(video_path)
    cropped = result["cropped"]

    if cropped.shape[0] < CHUNK_LENGTH:
        raise ValueError(
            f"Video too short: needs at least {CHUNK_LENGTH} frames at 30fps "
            f"(~{CHUNK_LENGTH/30:.1f}s), got {cropped.shape[0]} frames "
            f"(~{cropped.shape[0]/30:.1f}s)."
        )

    is_low_light = result["illumination"] < ILLUMINATION_THRESHOLD

    diff_normalized = diff_normalize_data(cropped)
    chunks = chunk_frames(diff_normalized, CHUNK_LENGTH)

    return {
        "chunks": chunks,
        "illumination": result["illumination"],
        "is_low_light": is_low_light,
        "face_found_ratio": result["face_found_ratio"],
        "n_frames": result["n_frames"],
        "fps": 30.0,
    }
