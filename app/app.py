import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emotion_detect_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow import keras

try:
    import mediapipe as mp
except Exception:
    mp = None


APP_TITLE = "Face Emotion Recognition"
WINDOW_SIZE = "1350x820"

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "Resnet50" / "face_emotion_resnet50_mediapipe_v3.keras"
METADATA_PATH = ROOT_DIR / "artifacts" / "models" / "Resnet50" / "metadata_resnet50_mediapipe_v3.json"
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

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

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_metadata(metadata_path=METADATA_PATH):
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


class ResNetEmotionApp:
    def __init__(self, root, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg="#f3f4f6")

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.class_names, self.img_size = load_metadata(self.metadata_path)

        self.model = None
        self.model_ready = False

        self.face_detector, self.face_detector_name = self._create_face_detector()

        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_result = None
        self.last_frame_time = time.perf_counter()
        self.flip_camera = False

        self.source_var = tk.StringVar(value="0")
        self.mode_var = tk.StringVar(value="realtime")
        self.mode_button_text = tk.StringVar(value="Switch to Static Image")
        self.min_confidence_var = tk.DoubleVar(value=0.5)
        self.face_margin_var = tk.DoubleVar(value=0.15)
        self.status_var = tk.StringVar(value="Loading ResNet50 model...")
        self.prediction_var = tk.StringVar(value="No prediction yet.")

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self._safe_load_model, daemon=True).start()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left_wrapper = ttk.Frame(main, width=400)
        left_wrapper.pack(side="left", fill="y", padx=(0, 10))
        left_wrapper.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_wrapper, highlightthickness=0)
        self.left_canvas.pack(side="left", fill="both", expand=True)

        left_scrollbar = ttk.Scrollbar(left_wrapper, orient="vertical", command=self.left_canvas.yview)
        left_scrollbar.pack(side="right", fill="y")
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)

        self.left_inner = ttk.Frame(self.left_canvas)
        self.left_window = self.left_canvas.create_window((0, 0), window=self.left_inner, anchor="nw")

        self.left_inner.bind("<Configure>", self._on_left_frame_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        mode_box = ttk.LabelFrame(self.left_inner, text="1) Mode", padding=10)
        mode_box.pack(fill="x", pady=(0, 10))

        self.mode_label = ttk.Label(mode_box, text="Current mode: Realtime camera")
        self.mode_label.pack(anchor="w", pady=(0, 6))
        ttk.Button(mode_box, textvariable=self.mode_button_text, command=self.toggle_mode).pack(fill="x")

        self.mode_content = ttk.Frame(self.left_inner)
        self.mode_content.pack(fill="x", pady=(0, 10))

        self.source_box = ttk.LabelFrame(self.mode_content, text="2) Camera / RTSP Source", padding=10)

        ttk.Label(self.source_box, text="RTSP URL or camera index (0, 1, ...)").pack(anchor="w")
        ttk.Entry(self.source_box, textvariable=self.source_var).pack(fill="x", pady=5)

        btns = ttk.Frame(self.source_box)
        btns.pack(fill="x", pady=5)

        ttk.Button(btns, text="Start Camera", command=self.start_camera).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(btns, text="Stop", command=self.stop_camera).pack(
            side="left", fill="x", expand=True, padx=(4, 4)
        )
        ttk.Button(btns, text="Flip Camera", command=self.toggle_flip_camera).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        self.image_box = ttk.LabelFrame(self.mode_content, text="2) Static Image", padding=10)
        ttk.Button(self.image_box, text="Choose Image", command=self.open_image).pack(fill="x")

        model_box = ttk.LabelFrame(self.left_inner, text="3) Model ResNet50", padding=10)
        model_box.pack(fill="x", pady=(0, 10))

        ttk.Label(model_box, text=f"Model: {self.model_path.name}").pack(anchor="w")
        ttk.Label(model_box, text=f"Metadata: {self.metadata_path.name}").pack(anchor="w", pady=(2, 8))
        ttk.Button(model_box, text="Reload Model", command=self.reload_model).pack(fill="x", pady=(0, 6))

        ttk.Label(model_box, text="Face crop margin").pack(anchor="w", pady=(8, 0))
        ttk.Scale(model_box, from_=0.0, to=0.35, variable=self.face_margin_var, orient="horizontal").pack(fill="x")

        ttk.Label(model_box, text="MediaPipe min confidence").pack(anchor="w", pady=(8, 0))
        ttk.Scale(model_box, from_=0.1, to=0.9, variable=self.min_confidence_var, orient="horizontal").pack(fill="x")
        ttk.Button(model_box, text="Reload Face Detector", command=self.reload_face_detector).pack(fill="x", pady=(8, 0))

        result_box = ttk.LabelFrame(self.left_inner, text="Current Result", padding=10)
        result_box.pack(fill="x", pady=(0, 10))
        ttk.Label(result_box, textvariable=self.prediction_var, wraplength=350, justify="left").pack(anchor="w")

        help_box = ttk.LabelFrame(self.left_inner, text="Instructions", padding=10)
        help_box.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Label(help_box, text=self.get_help_text(), wraplength=350, justify="left").pack(anchor="w")

        self.video_label = ttk.Label(right)
        self.video_label.pack(fill="both", expand=True)

        bottom = ttk.Frame(right)
        bottom.pack(fill="x", pady=(8, 0))

        self.info_text = tk.Text(bottom, height=8, wrap="word")
        self.info_text.pack(fill="x")
        self.info_text.insert("end", self.get_model_text())
        self.info_text.configure(state="disabled")

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        self.update_mode_controls()

    def _on_left_frame_configure(self, event):
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event):
        self.left_canvas.itemconfig(self.left_window, width=event.width)

    def _on_mousewheel(self, event):
        try:
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def get_help_text(self):
        return (
            "Demo workflow:\n"
            "1. Use the mode button to switch between realtime camera and static image prediction.\n"
            "2. In realtime mode, keep camera index as 0 for the default webcam.\n"
            "3. Click Start Camera to run realtime prediction.\n"
            "4. In static image mode, click Choose Image to run one prediction.\n"
            "5. If no detector is available, the app falls back to full-frame/full-image prediction."
        )

    def get_model_text(self):
        return (
            f"Model: ResNet50 emotion classifier\n"
            f"Input size: {self.img_size[0]} x {self.img_size[1]}\n"
            f"Classes: {', '.join(self.class_names)}\n"
            f"Face detector: {self.face_detector_name}\n"
            "Displayed output: label, confidence, top-3 probabilities, inference time, FPS."
        )

    def log_status(self, text):
        if hasattr(self, "status_var"):
            self.status_var.set(text)
            self.root.update_idletasks()

    def toggle_mode(self):
        if self.mode_var.get() == "realtime":
            self.mode_var.set("static")
            self.stop_camera()
            self.video_label.configure(image="")
            self.prediction_var.set("No image selected.")
            self.log_status("Static image mode selected.")
        else:
            self.mode_var.set("realtime")
            self.prediction_var.set("No prediction yet.")
            self.log_status("Realtime camera mode selected.")

        self.update_mode_controls()

    def update_mode_controls(self):
        self.source_box.pack_forget()
        self.image_box.pack_forget()

        if self.mode_var.get() == "realtime":
            self.mode_label.config(text="Current mode: Realtime camera")
            self.mode_button_text.set("Switch to Static Image")
            self.source_box.pack(fill="x")
        else:
            self.mode_label.config(text="Current mode: Static image")
            self.mode_button_text.set("Switch to Realtime")
            self.image_box.pack(fill="x")

    def _create_face_detector(self):
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=float(self.min_confidence_var.get())
                if hasattr(self, "min_confidence_var")
                else 0.5,
            )
            return detector, "MediaPipe"

        detector = cv2.CascadeClassifier(str(CASCADE_PATH))
        if not detector.empty():
            return detector, "OpenCV Haar Cascade"

        return None, "Full frame"

    def reload_face_detector(self):
        if hasattr(self.face_detector, "close"):
            self.face_detector.close()
        self.face_detector, self.face_detector_name = self._create_face_detector()
        self.log_status(f"Face detector reloaded: {self.face_detector_name}")

    def _safe_load_model(self):
        if not self.model_path.exists():
            self.model_ready = False
            self.log_status(f"Model not found: {self.model_path}")
            return

        start = time.perf_counter()
        try:
            self.model = keras.models.load_model(self.model_path)
            self.model_ready = True
            elapsed = time.perf_counter() - start
            self.log_status(f"ResNet50 model loaded in {elapsed:.1f}s.")
        except Exception as exc:
            self.model = None
            self.model_ready = False
            self.log_status(f"Cannot load model: {exc}")

    def reload_model(self):
        self.log_status("Reloading ResNet50 model...")
        threading.Thread(target=self._safe_load_model, daemon=True).start()

    def _open_capture(self, source_text):
        source_text = source_text.strip()
        source = int(source_text) if source_text.isdigit() else source_text

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
        return cap

    def start_camera(self):
        if self.mode_var.get() != "realtime":
            messagebox.showwarning("Wrong mode", "Switch to realtime mode before starting the camera.")
            return

        if self.running:
            self.log_status("Camera is already running.")
            return

        if not self.model_ready:
            messagebox.showwarning("Model not ready", "The ResNet50 model is still loading.")
            return

        cap = self._open_capture(self.source_var.get())
        if cap is None:
            messagebox.showerror("Error", "Cannot open camera / RTSP stream.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap = cap
        self.running = True
        self.last_frame_time = time.perf_counter()
        self.log_status("Camera started.")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="")
        self.log_status("Camera stopped.")

    def toggle_flip_camera(self):
        self.flip_camera = not self.flip_camera
        state = "ON" if self.flip_camera else "OFF"
        self.log_status(f"Camera flip is {state}.")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.log_status("Lost camera frame. Stopping camera.")
            self.stop_camera()
            return

        if self.flip_camera:
            frame = cv2.flip(frame, 1)

        self.current_frame = frame.copy()
        display = frame.copy()
        display, result_text = self.detect_and_predict(display, realtime=True)
        self.current_result = result_text
        self.prediction_var.set(result_text)

        self.show_frame(display)
        self.root.after(15, self.update_frame)

    def open_image(self):
        if self.mode_var.get() != "static":
            self.mode_var.set("static")
            self.stop_camera()
            self.update_mode_controls()

        if not self.model_ready:
            messagebox.showwarning("Model not ready", "The ResNet50 model is still loading.")
            return

        path = filedialog.askopenfilename(
            title="Choose Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            messagebox.showwarning("Unsupported file", "Choose a .jpg, .jpeg, .png, .bmp, or .webp image.")
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read the selected image.")
            return

        display, result_text = self.detect_and_predict(frame.copy(), realtime=False)
        self.current_frame = frame
        self.current_result = result_text
        self.prediction_var.set(f"{Path(path).name}\n{result_text}")
        self.show_frame(display)
        self.log_status(f"Image predicted: {Path(path).name}")

    def detect_and_predict(self, display_frame, realtime=False):
        if not self.model_ready or self.model is None:
            return display_frame, "Model is not ready."

        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_frame_time, 1e-6) if realtime else 0.0
        if realtime:
            self.last_frame_time = now

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image_h, image_w = display_frame.shape[:2]
        face_box, face_score = self.detect_face(display_frame, frame_rgb)

        if face_box is None:
            cv2.putText(
                display_frame,
                "No face detected",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            return display_frame, f"No face detected | Detector: {self.face_detector_name}"

        x, y, w, h = face_box
        x1, y1, x2, y2 = expand_box(
            x,
            y,
            w,
            h,
            image_w,
            image_h,
            float(self.face_margin_var.get()),
        )

        face_rgb = frame_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            return display_frame, "Invalid face crop."

        input_array = cv2.resize(face_rgb, self.img_size, interpolation=cv2.INTER_AREA)
        input_array = input_array.astype(np.float32)[None, ...]

        start = time.perf_counter()
        probs = self.model.predict(input_array, verbose=0)[0]
        inference_ms = (time.perf_counter() - start) * 1000.0

        pred_idx = int(np.argmax(probs))
        label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        top_indices = np.argsort(probs)[::-1][:3]
        top3 = ", ".join(
            f"{self.class_names[int(idx)]}: {float(probs[int(idx)]) * 100:.1f}%"
            for idx in top_indices
        )

        self.draw_prediction(display_frame, (x1, y1, x2, y2), f"{label} {confidence * 100:.1f}%")

        result = (
            f"Emotion: {label} ({confidence * 100:.1f}%)\n"
            f"Detector: {self.face_detector_name} | Face: {face_score * 100:.1f}%\n"
            f"Inference: {inference_ms:.1f} ms"
        )
        if realtime:
            result += f" | FPS: {fps:.1f}"
        result += f"\nTop 3: {top3}"
        self.log_status(result.replace("\n", " | "))
        return display_frame, result

    def detect_face(self, frame_bgr, frame_rgb):
        if self.face_detector_name == "MediaPipe":
            image_h, image_w = frame_bgr.shape[:2]
            result = self.face_detector.process(frame_rgb)
            if not result.detections:
                return None, 0.0

            best = max(result.detections, key=lambda item: item.score[0] if item.score else 0.0)
            box = best.location_data.relative_bounding_box
            x = int(box.xmin * image_w)
            y = int(box.ymin * image_h)
            w = int(box.width * image_w)
            h = int(box.height * image_h)
            score = float(best.score[0]) if best.score else 0.0
            return (x, y, w, h), score

        if self.face_detector_name == "Full frame":
            image_h, image_w = frame_bgr.shape[:2]
            return (0, 0, image_w, image_h), 0.0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(90, 90),
        )
        if len(faces) == 0:
            return None, 0.0

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return (int(x), int(y), int(w), int(h)), 1.0

    def draw_prediction(self, frame, box, text):
        x1, y1, x2, y2 = box
        color = (34, 197, 94)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
        label_y1 = max(0, y1 - text_h - baseline - 10)
        label_y2 = label_y1 + text_h + baseline + 10

        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 12, label_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 6, label_y2 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (10, 20, 16),
            2,
            cv2.LINE_AA,
        )

    def show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        w = max(self.video_label.winfo_width(), 800)
        h = max(self.video_label.winfo_height(), 550)
        img.thumbnail((w, h), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def on_close(self):
        try:
            self.left_canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass

        if hasattr(self.face_detector, "close"):
            self.face_detector.close()
        self.stop_camera()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Tkinter ResNet50 emotion app.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .keras model.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH, help="Path to metadata JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    ResNetEmotionApp(root, model_path=args.model, metadata_path=args.metadata)
    root.mainloop()


if __name__ == "__main__":
    main()
