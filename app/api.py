import base64
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emotion_detect_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from tensorflow import keras

try:
    import mediapipe as mp
    MP_IMPORT_ERROR = None
except Exception as exc:
    mp = None
    MP_IMPORT_ERROR = exc


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    "mobilenetv2": {
        "label": "MobileNetV2",
        "model_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "mobilenetv2"
        / "face_emotion_mobilenetv2_mediapipe_v2.keras",
        "metadata_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "mobilenetv2"
        / "metadata_mobilenetv2_mediapipe_v2.json",
    },
    "efficientnetb0": {
        "label": "EfficientNetB0",
        "model_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "EfficientNetB0"
        / "face_emotion_efficientnetb0_mediapipe.keras",
        "metadata_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "EfficientNetB0"
        / "metadata_efficientnetb0_mediapipe.json",
    },
    "resnet50": {
        "label": "ResNet50",
        "model_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "Resnet50"
        / "face_emotion_resnet50_mediapipe_v3.keras",
        "metadata_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "Resnet50"
        / "metadata_resnet50_mediapipe_v3.json",
    },
    "resemotenet": {
        "label": "ResEmoteNet",
        "model_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "Resemotenet"
        / "best_resemotenet_emotion_model.keras",
        "metadata_path": ROOT_DIR
        / "artifacts"
        / "models"
        / "Resemotenet"
        / "metadata_resemotenet_mediapipe_v2.json",
    },
}
DEFAULT_MODEL_NAME = "resnet50"
MODEL_PATH = MODEL_CONFIGS[DEFAULT_MODEL_NAME]["model_path"]
METADATA_PATH = MODEL_CONFIGS[DEFAULT_MODEL_NAME]["metadata_path"]
MEDIAPIPE_FACE_DETECTOR_PATH = (
    ROOT_DIR / "artifacts" / "models" / "mediapipe" / "blaze_face_short_range.tflite"
)

DEFAULT_CLASS_NAMES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class PredictBase64Request(BaseModel):
    image_base64: str = Field(..., description="Raw base64 image data or a data URL.")
    margin_ratio: float = Field(0.15, ge=0.0, le=0.5)
    model_name: str = Field(
        DEFAULT_MODEL_NAME,
        description="mobilenetv2, efficientnetb0, resnet50, or resemotenet",
    )


def load_metadata(metadata_path: Path = METADATA_PATH):
    if not metadata_path.exists():
        return DEFAULT_CLASS_NAMES, (224, 224)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata.get("class_names") or DEFAULT_CLASS_NAMES
    img_size = metadata.get("img_size", [224, 224])
    return list(class_names), (int(img_size[0]), int(img_size[1]))


def expand_box(x, y, w, h, image_w, image_h, margin_ratio):
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_w, x + w + margin_x)
    y2 = min(image_h, y + h + margin_y)
    return x1, y1, x2, y2


def decode_image(image_bytes: bytes):
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")

    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Use jpg, png, bmp, or webp.")
    return frame


def decode_base64_image(payload: str):
    data = payload.strip()
    if "," in data and data.lower().startswith("data:"):
        data = data.split(",", 1)[1]

    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.") from exc


def get_mediapipe_model_runtime_path():
    target_dir = Path(tempfile.gettempdir()) / "emotion_detect_mediapipe"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / MEDIAPIPE_FACE_DETECTOR_PATH.name
    if not target_path.exists() or target_path.stat().st_size != MEDIAPIPE_FACE_DETECTOR_PATH.stat().st_size:
        shutil.copy2(MEDIAPIPE_FACE_DETECTOR_PATH, target_path)
    return target_path


class EmotionPredictor:
    def __init__(
        self,
        default_model_name: str = DEFAULT_MODEL_NAME,
        min_detection_confidence: float = 0.5,
    ):
        self.default_model_name = default_model_name
        self.model_cache = {}
        self.detector = None
        self.detector_name = "MediaPipe"
        self.min_detection_confidence = min_detection_confidence

    def load(self):
        self.get_model_bundle(self.default_model_name)
        self.detector, self.detector_name = self._create_face_detector()

    def normalize_model_name(self, model_name: Optional[str]):
        name = (model_name or self.default_model_name).strip().lower()
        if name not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}'. Use one of: {', '.join(MODEL_CONFIGS)}",
            )
        return name

    def get_model_bundle(self, model_name: Optional[str]):
        name = self.normalize_model_name(model_name)
        if name in self.model_cache:
            return self.model_cache[name]

        config = MODEL_CONFIGS[name]
        model_path = Path(config["model_path"])
        metadata_path = Path(config["metadata_path"])
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")

        class_names, img_size = load_metadata(metadata_path)
        model = keras.models.load_model(model_path)
        bundle = {
            "name": name,
            "label": config["label"],
            "model_path": model_path,
            "metadata_path": metadata_path,
            "class_names": class_names,
            "img_size": img_size,
            "model": model,
        }
        self.model_cache[name] = bundle
        return bundle

    def _create_face_detector(self):
        if mp is None:
            raise RuntimeError(
                "MediaPipe is required for face detection. Install it with: python -m pip install mediapipe"
            ) from MP_IMPORT_ERROR

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.min_detection_confidence,
            )
            return detector, "MediaPipe Solutions"

        if not MEDIAPIPE_FACE_DETECTOR_PATH.exists():
            raise RuntimeError(
                f"MediaPipe Tasks face detector model not found: {MEDIAPIPE_FACE_DETECTOR_PATH}"
            )

        runtime_model_path = get_mediapipe_model_runtime_path()
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(runtime_model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=self.min_detection_confidence,
        )
        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        finally:
            os.chdir(original_cwd)
        return detector, "MediaPipe Tasks"

    def detect_faces(self, frame_bgr, frame_rgb):
        image_h, image_w = frame_bgr.shape[:2]

        if self.detector_name == "MediaPipe Solutions":
            result = self.detector.process(frame_rgb)
            if not result.detections:
                return []

            faces = []
            for detection in result.detections:
                box = detection.location_data.relative_bounding_box
                x1 = max(0, int(box.xmin * image_w))
                y1 = max(0, int(box.ymin * image_h))
                x2 = min(image_w, int((box.xmin + box.width) * image_w))
                y2 = min(image_h, int((box.ymin + box.height) * image_h))
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                if w == 0 or h == 0:
                    continue
                score = float(detection.score[0]) if detection.score else 0.0
                faces.append(
                    {
                        "box": [x1, y1, w, h],
                        "score": score,
                    }
                )
            return sorted(faces, key=lambda item: item["score"], reverse=True)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        if not result.detections:
            return []

        faces = []
        for detection in result.detections:
            box = detection.bounding_box
            x1 = max(0, int(box.origin_x))
            y1 = max(0, int(box.origin_y))
            x2 = min(image_w, int(box.origin_x + box.width))
            y2 = min(image_h, int(box.origin_y + box.height))
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue
            score = 0.0
            if getattr(detection, "categories", None):
                score = float(detection.categories[0].score)
            faces.append(
                {
                    "box": [x1, y1, w, h],
                    "score": score,
                }
            )
        return sorted(faces, key=lambda item: item["score"], reverse=True)

    def predict(self, frame_bgr, margin_ratio: float = 0.15, model_name: Optional[str] = DEFAULT_MODEL_NAME):
        bundle = self.get_model_bundle(model_name)
        model = bundle["model"]
        class_names = bundle["class_names"]
        img_size = bundle["img_size"]

        image_h, image_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detected_faces = self.detect_faces(frame_bgr, frame_rgb)
        if not detected_faces:
            return {
                "face_detected": False,
                "face_count": 0,
                "detector": self.detector_name,
                "model": {"name": bundle["name"], "label": bundle["label"]},
                "image": {"width": image_w, "height": image_h},
                "faces": [],
                "message": "No face detected.",
            }

        face_inputs = []
        face_payloads = []
        for index, face in enumerate(detected_faces, start=1):
            x, y, w, h = face["box"]
            x1, y1, x2, y2 = expand_box(x, y, w, h, image_w, image_h, margin_ratio)
            face_rgb = frame_rgb[y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue

            input_array = cv2.resize(face_rgb, img_size, interpolation=cv2.INTER_AREA)
            face_inputs.append(input_array.astype(np.float32))
            face_payloads.append(
                {
                    "index": index,
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "expanded_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "score": face["score"],
                }
            )

        if not face_inputs:
            raise HTTPException(status_code=422, detail="Invalid face crops.")

        batch = np.stack(face_inputs, axis=0)
        started = time.perf_counter()
        predictions = model.predict(batch, verbose=0)
        inference_ms = (time.perf_counter() - started) * 1000.0

        faces = []
        for face_payload, probs in zip(face_payloads, predictions):
            pred_idx = int(np.argmax(probs))
            top_indices = np.argsort(probs)[::-1][:3]
            face_payload["prediction"] = {
                "label": class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top3": [
                    {
                        "label": class_names[int(idx)],
                        "confidence": float(probs[int(idx)]),
                    }
                    for idx in top_indices
                ],
            }
            faces.append(face_payload)

        first_face = faces[0]

        return {
            "face_detected": True,
            "face_count": len(faces),
            "detector": self.detector_name,
            "model": {"name": bundle["name"], "label": bundle["label"]},
            "image": {"width": image_w, "height": image_h},
            "face": {
                "box": first_face["box"],
                "expanded_box": first_face["expanded_box"],
                "score": first_face["score"],
            },
            "faces": faces,
            "prediction": first_face["prediction"],
            "inference_ms": inference_ms,
        }


app = FastAPI(title="Face Emotion Detection", version="1.0.0")
predictor = EmotionPredictor()


@app.on_event("startup")
def startup_event():
    predictor.load()


@app.get("/")
def root():
    return HTMLResponse(
        """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Emotion Detection</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      color: #16202a;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      box-sizing: border-box;
    }
    main {
      width: min(980px, 100%);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: start;
    }
    section {
      background: #ffffff;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(22, 32, 42, 0.08);
    }
    h1 {
      grid-column: 1 / -1;
      margin: 0 0 2px;
      font-size: 28px;
      line-height: 1.2;
    }
    .subtitle {
      grid-column: 1 / -1;
      margin: 0 0 8px;
      color: #5a6875;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }
    input[type="file"], select {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #c9d3de;
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfd;
    }
    .model-label {
      margin-top: 12px;
    }
    button {
      width: 100%;
      margin-top: 12px;
      border: 0;
      border-radius: 6px;
      padding: 12px 14px;
      background: #1667c7;
      color: white;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .nav-link {
      display: inline-block;
      margin-top: 12px;
      border-radius: 6px;
      padding: 11px 14px;
      background: #52616f;
      color: #ffffff;
      font-size: 15px;
      font-weight: 700;
      text-decoration: none;
      text-align: center;
      box-sizing: border-box;
      width: 100%;
    }
    img {
      width: 100%;
      max-height: 430px;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid #d9e0e7;
      background: #eef2f6;
      display: none;
    }
    .result {
      display: grid;
      gap: 12px;
    }
    .metric {
      border-bottom: 1px solid #edf1f5;
      padding-bottom: 10px;
    }
    .metric:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .face-list {
      display: grid;
      gap: 8px;
    }
    .face-item {
      border: 1px solid #d9e0e7;
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfd;
      font-size: 14px;
      line-height: 1.45;
    }
    .label {
      color: #5a6875;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .value {
      font-size: 18px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f4f6f8;
      border-radius: 6px;
      padding: 12px;
      font-size: 13px;
      max-height: 260px;
      overflow: auto;
    }
    .error {
      color: #b42318;
      font-weight: 700;
    }
    @media (max-width: 760px) {
      body {
        padding: 14px;
        place-items: start center;
      }
      main {
        grid-template-columns: 1fr;
      }
      h1 {
        font-size: 24px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Face Emotion Detection</h1>
    <p class="subtitle">Upload ảnh khuôn mặt để nhận diện cảm xúc.</p>

    <section>
      <label for="imageInput">Chọn ảnh</label>
      <input id="imageInput" type="file" accept="image/*" />
      <label for="modelSelect" class="model-label">Model</label>
      <select id="modelSelect">
        <option value="resnet50" selected>ResNet50</option>
        <option value="resemotenet">ResEmoteNet</option>
        <option value="efficientnetb0">EfficientNetB0</option>
        <option value="mobilenetv2">MobileNetV2</option>
      </select>
      <button id="predictButton" type="button">Dự đoán cảm xúc</button>
      <a class="nav-link" href="/realtime">Open realtime camera</a>
      <p id="status" class="subtitle"></p>
      <img id="preview" alt="Ảnh đã chọn" />
    </section>

    <section class="result">
      <div class="metric">
        <div class="label">Faces found</div>
        <div id="faceCount" class="value">0</div>
      </div>
      <div class="metric">
        <div class="label">Cảm xúc</div>
        <div id="emotion" class="value">Chưa có kết quả</div>
      </div>
      <div class="metric">
        <div class="label">Độ tin cậy</div>
        <div id="confidence" class="value">-</div>
      </div>
      <div class="metric">
        <div class="label">Top 3</div>
        <div id="top3" class="value">-</div>
      </div>
      <div class="metric">
        <div class="label">All faces</div>
        <div id="faces" class="face-list">-</div>
      </div>
      <div class="metric">
        <div class="label">Thông tin khác</div>
        <pre id="raw">Kết quả JSON sẽ hiện ở đây.</pre>
      </div>
    </section>
  </main>

  <script>
    const imageInput = document.getElementById("imageInput");
    const modelSelect = document.getElementById("modelSelect");
    const predictButton = document.getElementById("predictButton");
    const preview = document.getElementById("preview");
    const statusEl = document.getElementById("status");
    const faceCountEl = document.getElementById("faceCount");
    const emotionEl = document.getElementById("emotion");
    const confidenceEl = document.getElementById("confidence");
    const facesEl = document.getElementById("faces");
    const top3El = document.getElementById("top3");
    const rawEl = document.getElementById("raw");

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (!file) {
        preview.style.display = "none";
        return;
      }
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      statusEl.textContent = "";
    });

    predictButton.addEventListener("click", async () => {
      const file = imageInput.files[0];
      if (!file) {
        statusEl.textContent = "Bạn cần chọn ảnh trước.";
        statusEl.className = "subtitle error";
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_name", modelSelect.value);

      predictButton.disabled = true;
      statusEl.className = "subtitle";
      statusEl.textContent = "Đang xử lý...";
      emotionEl.textContent = "Đang dự đoán";
      confidenceEl.textContent = "-";
      faceCountEl.textContent = "0";
      facesEl.textContent = "-";
      top3El.textContent = "-";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        rawEl.textContent = JSON.stringify(data, null, 2);

        if (!response.ok) {
          throw new Error(data.detail || "Không dự đoán được ảnh.");
        }
        if (!data.face_detected) {
          emotionEl.textContent = "Không tìm thấy khuôn mặt";
          confidenceEl.textContent = "-";
          faceCountEl.textContent = "0";
          facesEl.textContent = "-";
          top3El.textContent = "-";
          statusEl.textContent = data.message || "Không tìm thấy khuôn mặt.";
          return;
        }

        const faces = data.faces || [];
        faceCountEl.textContent = String(data.face_count || faces.length);
        emotionEl.textContent = data.prediction.label;
        confidenceEl.textContent = `${(data.prediction.confidence * 100).toFixed(1)}%`;
        top3El.textContent = data.prediction.top3
          .map((item) => `${item.label}: ${(item.confidence * 100).toFixed(1)}%`)
          .join(", ");
        facesEl.innerHTML = "";
        faces.forEach((face) => {
          const item = document.createElement("div");
          item.className = "face-item";
          const top3 = face.prediction.top3
            .map((entry) => `${entry.label} ${(entry.confidence * 100).toFixed(1)}%`)
            .join(", ");
          item.textContent = `Face ${face.index}: ${face.prediction.label} ${(face.prediction.confidence * 100).toFixed(1)}% | top3: ${top3}`;
          facesEl.appendChild(item);
        });
        statusEl.textContent = `Detector: ${data.detector} | faces: ${data.face_count} | ${data.inference_ms.toFixed(1)} ms`;
      } catch (error) {
        statusEl.className = "subtitle error";
        statusEl.textContent = error.message;
        emotionEl.textContent = "Lỗi";
      } finally {
        predictButton.disabled = false;
      }
    });
  </script>
</body>
</html>
        """
    )


@app.get("/realtime")
def realtime_page():
    return HTMLResponse(
        """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Realtime Face Emotion Detection</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      color: #16202a;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      box-sizing: border-box;
    }
    main {
      width: min(980px, 100%);
      display: grid;
      gap: 14px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }
    .subtitle {
      margin: 0;
      color: #5a6875;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      background: #1667c7;
      color: white;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    select {
      border: 1px solid #c9d3de;
      border-radius: 6px;
      padding: 10px 12px;
      background: #ffffff;
      color: #16202a;
      font-size: 15px;
      font-weight: 700;
    }
    button.secondary {
      background: #52616f;
    }
    .nav-link {
      display: inline-block;
      border-radius: 6px;
      padding: 11px 14px;
      background: #52616f;
      color: white;
      font-size: 15px;
      font-weight: 700;
      text-decoration: none;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .stage {
      background: #ffffff;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(22, 32, 42, 0.08);
    }
    canvas {
      width: 100%;
      max-height: 70vh;
      object-fit: contain;
      display: block;
      border-radius: 6px;
      background: #111820;
    }
    video {
      display: none;
    }
    .status {
      min-height: 22px;
      color: #5a6875;
      font-weight: 700;
    }
    .error {
      color: #b42318;
    }
    @media (max-width: 700px) {
      body {
        padding: 14px;
        place-items: start center;
      }
      h1 {
        font-size: 23px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Realtime Face Emotion Detection</h1>
    <p class="subtitle">Webcam realtime, MediaPipe face box, one highest-confidence emotion per face.</p>

    <div class="toolbar">
      <button id="startButton" type="button">Start camera</button>
      <button id="stopButton" class="secondary" type="button" disabled>Stop</button>
      <select id="modelSelect" aria-label="Model">
        <option value="resnet50" selected>ResNet50</option>
        <option value="resemotenet">ResEmoteNet</option>
        <option value="efficientnetb0">EfficientNetB0</option>
        <option value="mobilenetv2">MobileNetV2</option>
      </select>
      <a class="nav-link" href="/">Upload image</a>
      <span id="status" class="status">Camera is stopped.</span>
    </div>

    <div class="stage">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas" width="640" height="480"></canvas>
    </div>
  </main>

  <script>
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const modelSelect = document.getElementById("modelSelect");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const statusEl = document.getElementById("status");

    const emotionColors = {
      anger: "#e53935",
      contempt: "#8e24aa",
      disgust: "#43a047",
      fear: "#5e35b1",
      happy: "#f9a825",
      neutral: "#546e7a",
      sad: "#1e88e5",
      surprise: "#fb8c00",
    };

    let stream = null;
    let running = false;
    let busy = false;
    let lastPredictAt = 0;
    let latestFaces = [];
    let latestFaceCount = 0;
    const predictEveryMs = 450;

    function colorFor(label) {
      return emotionColors[String(label || "").toLowerCase()] || "#00acc1";
    }

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? "status error" : "status";
    }

    function drawFrame() {
      if (!running) {
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      latestFaces.forEach((face) => {
        const prediction = face.prediction;
        const label = prediction.label;
        const confidence = prediction.confidence;
        const color = colorFor(label);
        const box = face.box;

        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        const text = `${label} ${(confidence * 100).toFixed(1)}%`;
        ctx.font = "bold 18px Arial";
        const metrics = ctx.measureText(text);
        const labelHeight = 26;
        const labelY = Math.max(0, box.y - labelHeight);

        ctx.fillStyle = color;
        ctx.fillRect(box.x, labelY, metrics.width + 14, labelHeight);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, box.x + 7, labelY + 19);
      });

      const now = performance.now();
      if (!busy && now - lastPredictAt > predictEveryMs) {
        lastPredictAt = now;
        predictCurrentFrame();
      }

      requestAnimationFrame(drawFrame);
    }

    async function predictCurrentFrame() {
      busy = true;
      try {
        const imageBase64 = canvas.toDataURL("image/jpeg", 0.82);
        const response = await fetch("/predict-base64", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            image_base64: imageBase64,
            margin_ratio: 0.15,
            model_name: modelSelect.value,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Prediction failed.");
        }

        latestFaces = data.faces || [];
        latestFaceCount = data.face_count || latestFaces.length;
        if (latestFaceCount === 0) {
          setStatus("No face detected.");
        } else {
          setStatus(`Faces: ${latestFaceCount} | ${data.inference_ms.toFixed(1)} ms`);
        }
      } catch (error) {
        latestFaces = [];
        setStatus(error.message, true);
      } finally {
        busy = false;
      }
    }

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {width: 640, height: 480, facingMode: "user"},
          audio: false,
        });
        video.srcObject = stream;
        await video.play();

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        running = true;
        latestFaces = [];
        startButton.disabled = true;
        stopButton.disabled = false;
        setStatus("Camera is running.");
        requestAnimationFrame(drawFrame);
      } catch (error) {
        setStatus(`Cannot start camera: ${error.message}`, true);
      }
    }

    function stopCamera() {
      running = false;
      busy = false;
      latestFaces = [];
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      stream = null;
      startButton.disabled = false;
      stopButton.disabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setStatus("Camera is stopped.");
    }

    startButton.addEventListener("click", startCamera);
    stopButton.addEventListener("click", stopCamera);
    window.addEventListener("beforeunload", stopCamera);
  </script>
</body>
</html>
        """
    )


@app.get("/api")
def api_info():
    return {
        "name": "Face Emotion Detection API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }


@app.get("/models")
def list_models():
    return {
        "default": DEFAULT_MODEL_NAME,
        "loaded": sorted(predictor.model_cache.keys()),
        "models": [
            {
                "name": name,
                "label": config["label"],
                "model_path": str(config["model_path"]),
                "metadata_path": str(config["metadata_path"]),
                "available": Path(config["model_path"]).exists() and Path(config["metadata_path"]).exists(),
            }
            for name, config in MODEL_CONFIGS.items()
        ],
    }


@app.get("/health")
def health():
    default_bundle = predictor.get_model_bundle(DEFAULT_MODEL_NAME)
    return {
        "status": "ok" if predictor.model_cache else "model_not_loaded",
        "default_model": DEFAULT_MODEL_NAME,
        "loaded_models": sorted(predictor.model_cache.keys()),
        "model_path": str(default_bundle["model_path"]),
        "metadata_path": str(default_bundle["metadata_path"]),
        "classes": default_bundle["class_names"],
        "input_size": default_bundle["img_size"],
        "detector": predictor.detector_name,
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    margin_ratio: Optional[float] = Form(0.15),
    model_name: Optional[str] = Form(DEFAULT_MODEL_NAME),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload file must be an image.")

    image_bytes = await file.read()
    frame = decode_image(image_bytes)
    return predictor.predict(frame, margin_ratio=float(margin_ratio or 0.15), model_name=model_name)


@app.post("/predict-base64")
def predict_base64(request: PredictBase64Request):
    image_bytes = decode_base64_image(request.image_base64)
    frame = decode_image(image_bytes)
    return predictor.predict(frame, margin_ratio=request.margin_ratio, model_name=request.model_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
