"""
Flask web interface: upload a facial video, get a predicted heart rate.

Routes between the UBFC (normal-light) and MMPD fold3 (low-light) PhysNet
checkpoints based on measured face-region illumination in the uploaded clip.
"""
import os
import traceback
import uuid

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

from inference import estimate_heart_rate, load_models

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
# Render's free tier has only 512MB RAM total for the whole process (PyTorch
# runtime + both loaded checkpoints + Flask/gunicorn + the video itself while
# processing). A large upload could OOM-kill the instance before Flask even
# gets a chance to reject it gracefully, so this is deliberately much smaller
# than the 2GB used for local testing against big raw/uncompressed test
# clips. Real phone-recorded uploads are almost always well under this.
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", 150)) * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    # Ensure ALL errors (including Flask/Werkzeug's own, like 413 Payload
    # Too Large) come back as JSON instead of an HTML error page -- the
    # frontend always expects JSON and would otherwise fail with a cryptic
    # "Unexpected token '<'" parse error.
    return jsonify({"error": e.description or str(e)}), e.code


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    temp_name = f"{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    file.save(temp_path)

    try:
        result = estimate_heart_rate(temp_path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# Loaded at import time (not inside `if __name__ == "__main__"`) so this
# also runs under gunicorn/Render, which imports `app:app` directly and
# never executes the __main__ block. inference.py also lazy-loads on first
# request as a fallback, but eager-loading here avoids a slow first request.
print("Loading PhysNet checkpoints (normal-light + low-light)...")
load_models()
print("Models loaded.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
