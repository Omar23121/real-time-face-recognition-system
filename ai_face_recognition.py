import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


KNOWN_FACES_DIR = "known_faces"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "recognition_log.csv")
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "recognized_output.jpg")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "recognized_video.mp4")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"

# Lower = stricter
MATCH_THRESHOLD = 0.45

# Process every N frames for performance
PROCESS_EVERY_N_FRAMES = 3


def ensure_directories() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def ensure_log_file() -> None:
    ensure_directories()
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("timestamp,mode,label,distance,source\n")


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


def log_recognition(mode: str, label: str, distance: float, source: str) -> None:
    ensure_log_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_source = source.replace(",", "_")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{mode},{label},{distance:.4f},{safe_source}\n")


def cosine_distance(vec1, vec2) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0

    similarity = np.dot(a, b) / denom
    return float(1.0 - similarity)


def get_face_embedding(image_path: str):
    embedding_objs = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True
    )

    if not embedding_objs:
        return None

    return embedding_objs[0]["embedding"]


def build_known_faces_database():
    database = []

    if not os.path.exists(KNOWN_FACES_DIR):
        raise FileNotFoundError(f"Folder not found: {KNOWN_FACES_DIR}")

    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        loaded_count = 0

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(person_dir, filename)

            try:
                embedding = get_face_embedding(image_path)
                if embedding is None:
                    print(f"[WARNING] No face embedding created for: {image_path}")
                    continue

                database.append({
                    "label": person_name,
                    "path": image_path,
                    "embedding": embedding
                })
                loaded_count += 1

            except Exception as e:
                print(f"[WARNING] Failed processing {image_path}: {e}")

        print(f"[INFO] Loaded {loaded_count} face(s) for {person_name}")

    if not database:
        raise ValueError("No usable known face embeddings found in known_faces/")

    print(f"[INFO] Total known embeddings loaded: {len(database)}")
    return database


def detect_faces_in_frame(frame):
    faces = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=True
    )
    return faces


def get_embedding_from_face_crop(face_crop_bgr):
    embedding_objs = DeepFace.represent(
        img_path=face_crop_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False
    )

    if not embedding_objs:
        return None

    return embedding_objs[0]["embedding"]


def normalize_box(face_dict: dict, image_width: int, image_height: int):
    area = face_dict.get("facial_area", {})
    x = max(0, int(area.get("x", 0)))
    y = max(0, int(area.get("y", 0)))
    w = int(area.get("w", 0))
    h = int(area.get("h", 0))

    x2 = min(image_width - 1, x + w)
    y2 = min(image_height - 1, y + h)

    return x, y, x2, y2


def match_embedding_to_database(face_embedding, database):
    best_match = None
    best_distance = float("inf")

    for item in database:
        dist = cosine_distance(face_embedding, item["embedding"])

        if dist < best_distance:
            best_distance = dist
            best_match = item

    if best_match is None:
        return {
            "label": "Unknown",
            "distance": 1.0,
            "path": None
        }

    if best_distance > MATCH_THRESHOLD:
        return {
            "label": "Unknown",
            "distance": best_distance,
            "path": None
        }

    return {
        "label": best_match["label"],
        "distance": best_distance,
        "path": best_match["path"]
    }


def annotate_faces_on_frame(frame, database, mode: str, source: str):
    frame_height, frame_width = frame.shape[:2]

    try:
        faces = detect_faces_in_frame(frame)
    except Exception as e:
        print(f"[WARNING] Face detection failed: {e}")
        faces = []

    results = []

    for face in faces:
        x1, y1, x2, y2 = normalize_box(face, frame_width, frame_height)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        try:
            embedding = get_embedding_from_face_crop(face_crop)
            if embedding is None:
                match = {"label": "Unknown", "distance": 1.0, "path": None}
            else:
                match = match_embedding_to_database(embedding, database)
        except Exception as e:
            print(f"[WARNING] Failed matching one face: {e}")
            match = {"label": "Unknown", "distance": 1.0, "path": None}

        label = match["label"]
        distance = match["distance"]

        log_recognition(mode, label, distance, source)

        results.append({
            "box": (x1, y1, x2, y2),
            "label": label,
            "distance": distance
        })

    return results


def draw_results(frame, results, fps=None):
    output = frame.copy()

    for item in results:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        distance = item["distance"]

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        display_label = format_label(label)
        text = f"{display_label} | dist={distance:.3f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if y1 - 10 > 10 else y2 + 25
        cv2.putText(
            output,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    if fps is not None:
        cv2.putText(
            output,
            f"FPS: {fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

    return output


def image_mode(image_path: str, database) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = annotate_faces_on_frame(image, database, mode="image", source=image_path)
    final_image = draw_results(image, results)

    ensure_directories()
    cv2.imwrite(OUTPUT_IMAGE, final_image)

    print(f"[INFO] Faces detected: {len(results)}")
    for item in results:
        print(f"[INFO] Match: {item['label']} | dist={item['distance']:.4f}")

    print(f"[INFO] Saved result to {OUTPUT_IMAGE}")
    print(f"[INFO] Logs saved to {LOG_FILE}")


def webcam_mode(database):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Webcam started.")
    print("[INFO] Press 'q' to quit.")

    frame_count = 0
    cached_results = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam.")
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            cached_results = annotate_faces_on_frame(
                frame,
                database,
                mode="webcam",
                source="webcam"
            )

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        display_frame = draw_results(frame, cached_results, fps=fps)
        cv2.imshow("AI Face Recognition Webcam", display_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Logs saved to {LOG_FILE}")


def video_mode(video_path: str, database):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    ensure_directories()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (width, height))

    print("[INFO] Video mode started.")
    print("[INFO] Press 'q' to stop early.")

    frame_count = 0
    cached_results = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            cached_results = annotate_faces_on_frame(
                frame,
                database,
                mode="video",
                source=video_path
            )

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        display_frame = draw_results(frame, cached_results, fps=fps)
        writer.write(display_frame)
        cv2.imshow("AI Face Recognition Video", display_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Saved annotated video to {OUTPUT_VIDEO}")
    print(f"[INFO] Logs saved to {LOG_FILE}")


def print_usage():
    print("Usage:")
    print("  python ai_face_recognition.py image <image_path>")
    print("  python ai_face_recognition.py webcam")
    print("  python ai_face_recognition.py video <video_path>")


if __name__ == "__main__":
    ensure_directories()
    ensure_log_file()

    try:
        database = build_known_faces_database()
    except Exception as e:
        print(f"[ERROR] Failed to build known faces database: {e}")
        sys.exit(1)

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "image":
        if len(sys.argv) < 3:
            print("[ERROR] Missing image path.")
            print_usage()
            sys.exit(1)

        image_path = sys.argv[2]

        try:
            image_mode(image_path, database)
        except Exception as e:
            print(f"[ERROR] Image mode failed: {e}")
            sys.exit(1)

    elif mode == "webcam":
        try:
            webcam_mode(database)
        except Exception as e:
            print(f"[ERROR] Webcam mode failed: {e}")
            sys.exit(1)

    elif mode == "video":
        if len(sys.argv) < 3:
            print("[ERROR] Missing video path.")
            print_usage()
            sys.exit(1)

        video_path = sys.argv[2]

        try:
            video_mode(video_path, database)
        except Exception as e:
            print(f"[ERROR] Video mode failed: {e}")
            sys.exit(1)

    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print_usage()
        sys.exit(1)