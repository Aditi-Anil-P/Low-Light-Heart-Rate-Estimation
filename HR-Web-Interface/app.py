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
# Some raw/uncompressed test clips (e.g. UBFC-rPPG .avi files) can be 1GB+;
# real-world phone-recorded uploads are typically much smaller once
# compressed, but this is set generously for local testing.
MAX_CONTENT_LENGTH = 2048 * 1024 * 1024  # 2 GB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    # Ensure ALL errors (including Flask/Werkzeug's own, like 413 Payload
    # Too Large) come back as JSON instead of an HTML error page -- the
    # frontend always expects JSON and would otherwise fail with a cryptic
    # "Unexpected token '<'" parse error.
    return jsonify({"error": e.description or str(e)}), e.code


@app.errorhandler(Exception)
def handle_uncaught_exception(e):
    # Safety net for anything NOT already caught by predict()'s own
    # try/except -- e.g. exceptions raised by Werkzeug while parsing a large
    # multipart upload (request.files access, before our view body even
    # runs). Without this, such errors fell through to Flask's default
    # handler, which for a non-HTTPException replaces the real message with
    # a generic "The server encountered an internal error..." page -- that's
    # what large-file uploads were showing, with the actual cause hidden.
    if isinstance(e, HTTPException):
        return handle_http_exception(e)
    traceback.print_exc()
    return jsonify({"error": str(e) or type(e).__name__}), 500


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

    try:
        # file.save() included here too -- for very large uploads this can
        # itself fail (disk space, memory) before estimate_heart_rate ever
        # runs, and was previously outside this try/except, surfacing as a
        # generic unreadable "Internal Server Error" page instead of JSON.
        file.save(temp_path)
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


if __name__ == "__main__":
    print("Loading PhysNet checkpoints (normal-light + low-light)...")
    load_models()
    print("Models loaded. Starting server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
