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


# Hard cap on frames read into memory, regardless of the uploaded file's
# actual length. Whole-video concatenation gives diminishing HR-accuracy
# returns well before this (our own testing used ~60s clips), but a
# resource-constrained deploy (e.g. Render's free 512MB RAM tier) can't
# afford to load an arbitrarily long/high-res video into a single array.
# At 30fps this is 60s of footage -- comfortably more than needed.
MAX_SOURCE_FRAMES = 1800


def read_video(video_path):
    """Read an mp4/avi/etc. video file into an array of RGB frames [T,H,W,3]
    (uint8), capped at MAX_SOURCE_FRAMES to bound memory usage."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < MAX_SOURCE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames could be read from the video.")
    return np.asarray(frames, dtype=np.uint8), fps


def resample_to_30fps(frames, src_fps):
    """Resample a frame sequence to 30fps by nearest-index selection.

    The checkpoints were trained on 30fps data (UBFC/MMPD). Arbitrary uploads
    may come in at other frame rates, so we resample the frame index grid
    rather than assuming 30fps.
    """
    target_fps = 30.0
    if src_fps is None or src_fps <= 0 or abs(src_fps - target_fps) < 0.5:
        return frames
    n_src = frames.shape[0]
    duration_sec = n_src / src_fps
    n_target = max(1, int(round(duration_sec * target_fps)))
    src_indices = np.linspace(0, n_src - 1, num=n_target)
    src_indices = np.round(src_indices).astype(int)
    src_indices = np.clip(src_indices, 0, n_src - 1)
    return frames[src_indices]


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


def estimate_illumination(frames, sample_stride=5):
    """Estimate mean grayscale intensity within the detected face region,
    averaged over a sample of frames. Returns (mean_intensity, face_found_ratio).
    Falls back to whole-frame mean intensity if no face is ever detected.
    """
    n = frames.shape[0]
    sample_idx = range(0, n, max(1, sample_stride))
    face_vals = []
    whole_vals = []
    faces_found = 0
    checked = 0
    for i in sample_idx:
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        whole_vals.append(gray.mean())
        checked += 1
        box = detect_largest_face(frame)
        if box is not None:
            x, y, w, h = enlarge_box(box, frame.shape)
            face_crop = gray[y:y + h, x:x + w]
            if face_crop.size > 0:
                face_vals.append(face_crop.mean())
                faces_found += 1
    face_found_ratio = faces_found / max(1, checked)
    if len(face_vals) > 0:
        return float(np.mean(face_vals)), face_found_ratio
    return float(np.mean(whole_vals)), face_found_ratio


def crop_resize_frames(frames, use_larger_box=True):
    """Detect the largest face on the first frame (static detection, matching
    BaseLoader's default dynamic_detection=False), then crop and resize every
    frame in the clip to FACE_SIZE x FACE_SIZE using that single box.
    """
    box = detect_largest_face(frames[0])
    if box is None:
        H, W = frames[0].shape[:2]
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        box = (x0, y0, side, side)
    if use_larger_box:
        box = enlarge_box(box, frames[0].shape)
    x, y, w, h = box
    w = max(1, w)
    h = max(1, h)

    # NOTE: must be float, not uint8. diff_normalize_data() below computes
    # frame-to-frame sums and differences; uint8 arithmetic silently wraps
    # around (e.g. 10-250 -> 16 instead of -240), corrupting the signal.
    # The reference BaseLoader.crop_face_resize() uses a float64 buffer for
    # exactly this reason.
    out = np.zeros((frames.shape[0], FACE_SIZE, FACE_SIZE, 3), dtype=np.float64)
    for i in range(frames.shape[0]):
        crop = frames[i, y:y + h, x:x + w, :]
        if crop.size == 0:
            crop = frames[i]
        out[i] = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    return out


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
    """Full pipeline: read -> resample to 30fps -> illumination check ->
    face crop/resize -> diff-normalize -> chunk into model-ready tensors.
    """
    frames, src_fps = read_video(video_path)
    frames = resample_to_30fps(frames, src_fps)

    if frames.shape[0] < CHUNK_LENGTH:
        raise ValueError(
            f"Video too short: needs at least {CHUNK_LENGTH} frames at 30fps "
            f"(~{CHUNK_LENGTH/30:.1f}s), got {frames.shape[0]} frames "
            f"(~{frames.shape[0]/30:.1f}s)."
        )

    illumination, face_found_ratio = estimate_illumination(frames)
    is_low_light = illumination < ILLUMINATION_THRESHOLD

    cropped = crop_resize_frames(frames)
    diff_normalized = diff_normalize_data(cropped)
    chunks = chunk_frames(diff_normalized, CHUNK_LENGTH)

    return {
        "chunks": chunks,
        "illumination": illumination,
        "is_low_light": is_low_light,
        "face_found_ratio": face_found_ratio,
        "n_frames": int(frames.shape[0]),
        "fps": 30.0,
    }
