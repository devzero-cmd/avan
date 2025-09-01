import cv2
import numpy as np
import pyttsx3
from collections import deque
from ultralytics import YOLO
import time

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "yolov8n.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
HISTORY_LEN = 8
SPEECH_INTERVAL = 1.0
AREA_THRESHOLD_MIN = 0.02
AREA_THRESHOLD_BIG = 0.12
WALL_THRESHOLD = 0.35  # object covering >35% of view considered a wall
COVERAGE_RATIO_THRESHOLD = 0.55  # NEW: >55% of frame covered = wall
DEBUG = True

OBSTACLE_CLASSES = {
    "door",
    "stairs",
    "wall",
    "person",
    "vehicle",
    "fence",
    "pole",
    "chair",
    "sofa",
    "bench",
    "couch",
    "table",
    "bed",
    "bicycle",
    "motorbike",
    "trash can",
    "bag",
    "backpack",
}
INDOOR_CLASSES = {"door", "stairs", "wall", "person", "chair", "sofa", "bed"}
OUTDOOR_CLASSES = {"car", "truck", "bus", "road", "sidewalk", "fence", "pole", "tree"}

# -------------------------------
# INIT
# -------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera (index 0).")

model = YOLO(MODEL_PATH)

history = deque(maxlen=HISTORY_LEN)
last_msg = ""
last_speech_time = 0
block_count = 0


# -------------------------------
# SPEAK FUNCTION
# -------------------------------
def speak(msg: str):
    global last_msg, last_speech_time
    now = time.time()
    if msg != last_msg or (now - last_speech_time) > SPEECH_INTERVAL:
        try:
            engine.stop()
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
        last_msg = msg
        last_speech_time = now


# -------------------------------
# DRAW HELPERS
# -------------------------------
def draw_guidance(frame, msg, safe_dir):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, msg, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cx, cy = w // 2, h // 2
    if safe_dir == "left":
        pts = np.array(
            [[cx - 60, cy], [cx + 10, cy - 40], [cx + 10, cy + 40]], np.int32
        )
        cv2.fillPoly(frame, [pts], (0, 255, 255))
    elif safe_dir == "right":
        pts = np.array(
            [[cx + 60, cy], [cx - 10, cy - 40], [cx - 10, cy + 40]], np.int32
        )
        cv2.fillPoly(frame, [pts], (0, 255, 255))
    else:
        pts = np.array(
            [[cx, cy - 60], [cx - 40, cy + 10], [cx + 40, cy + 10]], np.int32
        )
        cv2.fillPoly(frame, [pts], (0, 255, 255))


# -------------------------------
# MAIN LOOP
# -------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Flip camera if inverted
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        crop = frame[h // 4 : 3 * h // 4, :]
        crop_h, crop_w = crop.shape[:2]
        frame_area = crop_h * crop_w

        results = model.predict(source=crop, imgsz=IMG_SIZE, verbose=False)

        status = ["clear", "clear", "clear"]
        indoor_detected = False
        outdoor_detected = False
        detections = 0
        wall_detected = False
        debug_boxes = []

        total_covered_area = 0.0  # NEW coverage area counter

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                conf_val = float(conf)
                if conf_val < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box]
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                box_area = box_w * box_h
                area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                total_covered_area += box_area  # accumulate covered area

                label = model.names[int(cls)].lower().replace("_", " ")

                if area_ratio < AREA_THRESHOLD_MIN:
                    debug_boxes.append((x1, y1, x2, y2, label, conf_val, False))
                    continue

                detections += 1
                debug_boxes.append((x1, y1, x2, y2, label, conf_val, True))

                if label in INDOOR_CLASSES:
                    indoor_detected = True
                elif label in OUTDOOR_CLASSES:
                    outdoor_detected = True

                x_center = (x1 + x2) / 2
                if x_center < crop_w / 3:
                    zone = 0
                elif x_center < 2 * crop_w / 3:
                    zone = 1
                else:
                    zone = 2

                if label in OBSTACLE_CLASSES or area_ratio >= AREA_THRESHOLD_BIG:
                    status[zone] = "blocked"

                # --- Wall check (ignore person) ---
                if area_ratio >= WALL_THRESHOLD and zone == 1 and label != "person":
                    wall_detected = True

        # NEW: compute coverage ratio
        coverage_ratio = total_covered_area / frame_area if frame_area > 0 else 0.0
        if coverage_ratio > COVERAGE_RATIO_THRESHOLD:
            wall_detected = True

        # Smooth with history
        history.append(status.copy())
        hist_array = np.array(
            [[1 if s == "blocked" else 0 for s in f] for f in history]
        )
        smoothed = (
            np.mean(hist_array, axis=0) if hist_array.size else np.array([0, 0, 0])
        )
        status = ["blocked" if s > 0.5 else "clear" for s in smoothed]

        msg = "Path clear"
        safe_dir = "forward"

        if wall_detected or (
            status[0] == "blocked" and status[1] == "blocked" and status[2] == "blocked"
        ):
            msg = "Wall ahead. Stop. Please turn left and right."
            if status[0] == "blocked" and status[2] == "clear":
                msg = "Right side blocked. Please move left."
                safe_dir = "left"
            elif status[2] == "blocked" and status[0] == "clear":
                msg = "Left side blocked. Please move right."
                safe_dir = "right"
            elif status[0] == "blocked" and status[2] == "blocked":
                msg = "No opening. Please move backward slowly."
        else:
            if status[1] == "blocked":
                if status[0] == "clear":
                    msg = "Center blocked, move left"
                    safe_dir = "left"
                elif status[2] == "clear":
                    msg = "Center blocked, move right"
                    safe_dir = "right"
                else:
                    block_count += 1
                    msg = (
                        "Stop, obstacle unavoidable"
                        if block_count > 3
                        else "Keep forward"
                    )
            elif status[0] == "blocked" and status[2] == "clear":
                msg = "Left blocked, keep right"
                safe_dir = "right"
                block_count = 0
            elif status[2] == "blocked" and status[0] == "clear":
                msg = "Right blocked, keep left"
                safe_dir = "left"
                block_count = 0
            else:
                block_count = 0

        if indoor_detected and not outdoor_detected:
            msg += " (Indoor)"
        elif outdoor_detected and not indoor_detected:
            msg += " (Outdoor)"
        else:
            msg += " (Mixed)"

        print(msg)
        speak(msg)

        if DEBUG:
            cv2.rectangle(frame, (0, h // 4), (w, 3 * h // 4), (50, 50, 50), 2)
            for x1, y1, x2, y2, label, conf_val, is_blocker in debug_boxes:
                x1i, y1i, x2i, y2i = (
                    int(x1),
                    int(y1) + h // 4,
                    int(x2),
                    int(y2) + h // 4,
                )
                color = (0, 0, 255) if is_blocker else (255, 165, 0)
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf_val:.2f}",
                    (x1i, y1i - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            draw_guidance(frame, msg, safe_dir)
            cv2.imshow("NAV DEBUG", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
