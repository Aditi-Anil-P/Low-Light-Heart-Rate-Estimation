(() => {
  "use strict";

  /* ---------------- config ---------------- */

  const ALLOWED_EXT = ["mp4", "avi", "mov", "mkv", "webm"];
  const MAX_BYTES = 2048 * 1024 * 1024; // 2GB, mirrors app.py
  const MIN_DURATION_S = 4.3; // 128 frames @ 30fps, mirrors preprocess.py

  // Staged, *approximate* progress. The backend is a single synchronous
  // request/response with no real progress channel, so instead of faking a
  // percentage we advance honest, labeled stages on a timer and jump straight
  // to "done" the moment the real response lands (whichever comes first).
  const PROCESSING_STEPS = [
    { delay: 0 },
    { delay: 900 },
    { delay: 2200 },
    { delay: 3600 },
    { delay: 6500 },
  ];

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- element refs ---------------- */

  const panel = document.getElementById("panel");
  const states = {
    upload: document.getElementById("state-upload"),
    preview: document.getElementById("state-preview"),
    processing: document.getElementById("state-processing"),
    result: document.getElementById("state-result"),
    error: document.getElementById("state-error"),
  };

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const uploadError = document.getElementById("uploadError");

  const previewVideo = document.getElementById("previewVideo");
  const previewName = document.getElementById("previewName");
  const previewDetails = document.getElementById("previewDetails");
  const previewWarning = document.getElementById("previewWarning");
  const changeFileBtn = document.getElementById("changeFileBtn");
  const submitBtn = document.getElementById("submitBtn");

  const cancelBtn = document.getElementById("cancelBtn");
  const stepEls = Array.from(document.querySelectorAll("#processingSteps li"));

  const routeBadge = document.getElementById("routeBadge");
  const bpmNumber = document.getElementById("bpmNumber");
  const illuminationFill = document.getElementById("illuminationFill");
  const illuminationValue = document.getElementById("illuminationValue");
  const faceRatioFill = document.getElementById("faceRatioFill");
  const faceRatioValue = document.getElementById("faceRatioValue");
  const framesValue = document.getElementById("framesValue");
  const fullClipValue = document.getElementById("fullClipValue");
  const chunksList = document.getElementById("chunksList");
  const copyBtn = document.getElementById("copyBtn");
  const restartBtn = document.getElementById("restartBtn");

  const errorTitle = document.getElementById("errorTitle");
  const errorDetail = document.getElementById("errorDetail");
  const retryBtn = document.getElementById("retryBtn");

  /* ---------------- state ---------------- */

  let selectedFile = null;
  let objectUrl = null;
  let stepTimers = [];
  let abortController = null;
  let lastResult = null;

  function showState(name) {
    Object.entries(states).forEach(([key, el]) => {
      el.hidden = key !== name;
    });
  }

  /* ---------------- file selection & validation ---------------- */

  function extOf(filename) {
    const parts = filename.split(".");
    return parts.length > 1 ? parts.pop().toLowerCase() : "";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KB", "MB", "GB"];
    let val = bytes;
    let i = -1;
    do {
      val /= 1024;
      i++;
    } while (val >= 1024 && i < units.length - 1);
    return `${val.toFixed(1)} ${units[i]}`;
  }

  function formatDuration(seconds) {
    if (!isFinite(seconds)) return "unknown length";
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  function rejectFile(message) {
    uploadError.textContent = message;
    uploadError.hidden = false;
  }

  function handleFile(file) {
    uploadError.hidden = true;

    const ext = extOf(file.name);
    if (!ALLOWED_EXT.includes(ext)) {
      rejectFile(`Unsupported file type ".${ext || "?"}". Allowed: ${ALLOWED_EXT.join(", ")}.`);
      return;
    }
    if (file.size > MAX_BYTES) {
      rejectFile(`File is ${formatBytes(file.size)} — the server accepts up to 2 GB.`);
      return;
    }

    selectedFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);

    previewVideo.src = objectUrl;
    previewName.textContent = file.name;
    previewDetails.textContent = `${formatBytes(file.size)} · reading length…`;
    previewWarning.hidden = true;

    previewVideo.onloadedmetadata = () => {
      const dur = previewVideo.duration;
      previewDetails.textContent = `${formatBytes(file.size)} · ${formatDuration(dur)}`;
      if (isFinite(dur) && dur < MIN_DURATION_S) {
        previewWarning.hidden = false;
        previewWarning.textContent =
          `This clip is shorter than the ~${MIN_DURATION_S}s minimum (128 frames at 30fps) — the server will likely reject it.`;
      }
    };

    showState("preview");
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  changeFileBtn.addEventListener("click", () => {
    resetToUpload();
  });

  function resetToUpload() {
    selectedFile = null;
    fileInput.value = "";
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    uploadError.hidden = true;
    showState("upload");
  }

  /* ---------------- processing / staged steps ---------------- */

  function clearStepTimers() {
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
  }

  function setActiveStep(index) {
    stepEls.forEach((el, i) => {
      el.classList.toggle("active", i === index);
      el.classList.toggle("done", i < index);
    });
  }

  function startProcessingUI() {
    setActiveStep(0);
    PROCESSING_STEPS.forEach((step, i) => {
      if (i === 0) return;
      const t = setTimeout(() => setActiveStep(i), step.delay);
      stepTimers.push(t);
    });
  }

  /* ---------------- submit ---------------- */

  submitBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    showState("processing");
    startProcessingUI();

    const formData = new FormData();
    formData.append("video", selectedFile);
    abortController = new AbortController();

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });
      const data = await res.json();
      clearStepTimers();

      if (!res.ok || data.error) {
        showError(data.error || "Something went wrong on the server.");
        return;
      }
      showResult(data);
    } catch (err) {
      clearStepTimers();
      if (err.name === "AbortError") {
        resetToUpload();
        return;
      }
      showError(`Request failed: ${err.message || err}. Check that the server is still running.`);
    }
  });

  cancelBtn.addEventListener("click", () => {
    clearStepTimers();
    if (abortController) abortController.abort();
  });

  /* ---------------- result rendering ---------------- */

  function showResult(data) {
    lastResult = data;

    const isLowLight = /low/i.test(data.route || "");
    routeBadge.textContent = data.route || "unknown route";
    routeBadge.classList.toggle("low-light", isLowLight);
    routeBadge.classList.toggle("normal-light", !isLowLight);

    bpmNumber.textContent = data.bpm ?? "--";

    const illum = Number(data.illumination) || 0;
    illuminationFill.style.width = `${Math.min(100, (illum / 255) * 100)}%`;
    illuminationValue.textContent = `${data.illumination} / 255`;

    const faceRatio = Number(data.face_found_ratio) || 0;
    faceRatioFill.style.width = `${Math.min(100, faceRatio * 100)}%`;
    faceRatioValue.textContent = `${Math.round(faceRatio * 100)}%`;

    framesValue.textContent = `${data.n_frames} frames · ${data.n_chunks} chunk${data.n_chunks === 1 ? "" : "s"}`;
    fullClipValue.textContent = `${data.bpm_full_clip} BPM`;

    chunksList.innerHTML = "";
    (data.per_chunk_bpm || []).forEach((bpm) => {
      const chip = document.createElement("span");
      chip.className = "chunk-chip";
      chip.textContent = `${bpm}`;
      chunksList.appendChild(chip);
    });

    showState("result");
  }

  copyBtn.addEventListener("click", async () => {
    if (!lastResult) return;
    const summary =
      `${lastResult.bpm} BPM (${lastResult.route})\n` +
      `Illumination: ${lastResult.illumination}/255 · Face detected ${Math.round((lastResult.face_found_ratio || 0) * 100)}%\n` +
      `Full-clip estimate: ${lastResult.bpm_full_clip} BPM · Per-chunk: ${(lastResult.per_chunk_bpm || []).join(", ")}`;
    try {
      await navigator.clipboard.writeText(summary);
      const original = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = original), 1500);
    } catch {
      /* clipboard API unavailable — silently ignore, not critical */
    }
  });

  restartBtn.addEventListener("click", resetToUpload);

  /* ---------------- error rendering ---------------- */

  // Light friendliness pass over common backend error strings. Falls back to
  // the raw message untouched — never invents detail the server didn't give.
  function friendlyError(raw) {
    const msg = String(raw || "");
    if (/too short|at least \d+ frames|128 frames/i.test(msg)) {
      return { title: "Clip is too short", detail: msg };
    }
    if (/no face|face.*not.*detect/i.test(msg)) {
      return { title: "No face detected", detail: msg };
    }
    if (/unsupported file type/i.test(msg)) {
      return { title: "Unsupported file type", detail: msg };
    }
    if (/payload too large|413/i.test(msg)) {
      return { title: "File too large", detail: msg };
    }
    return { title: "Something went wrong", detail: msg || "No further detail was returned." };
  }

  function showError(raw) {
    const { title, detail } = friendlyError(raw);
    errorTitle.textContent = title;
    errorDetail.textContent = detail;
    showState("error");
  }

  retryBtn.addEventListener("click", () => {
    if (selectedFile) {
      showState("preview");
    } else {
      resetToUpload();
    }
  });

  /* ---------------- ambient waveform animation ---------------- */
  // Draws a lightweight synthetic pulse trace on an SVG polyline. Used for
  // (a) the idle header divider and (b) the active "we're working on it"
  // indicator during processing — same visual language, different speed.

  function animateWave(pathEl, { width, height, points, speed, amplitude }) {
    if (reduceMotion) return; // static line already set in markup
    let t = 0;
    function frame() {
      const coords = [];
      for (let x = 0; x <= width; x += width / points) {
        const y =
          height / 2 +
          Math.sin(x * 0.04 + t) * amplitude * 0.4 +
          Math.sin(x * 0.11 + t * 1.7) * amplitude * 0.6 * pulseEnvelope(x, width, t);
        coords.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      }
      pathEl.setAttribute("points", coords.join(" "));
      t += speed;
      requestAnimationFrame(frame);
    }
    frame();
  }

  // Produces an occasional sharp "heartbeat" spike rather than a pure sine,
  // so the trace reads as a pulse rather than generic decoration.
  function pulseEnvelope(x, width, t) {
    const cycle = ((x / width) * 6 + t * 2.4) % 6;
    if (cycle > 2.7 && cycle < 3.3) return 3.2;
    return 1;
  }

  animateWave(document.getElementById("heroWavePath"), {
    width: 800, height: 60, points: 160, speed: 0.02, amplitude: 6,
  });
  animateWave(document.getElementById("pulseWavePath"), {
    width: 400, height: 100, points: 120, speed: 0.05, amplitude: 16,
  });
})();