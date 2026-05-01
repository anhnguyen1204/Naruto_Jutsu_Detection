"""
GNN Step 1: Collect hand landmark data for GNN training.
- Detects up to 2 hands; concatenates both (42 nodes total)
- If only 1 hand detected, the second hand is zero-padded
- Saves 42 × 3 = 126 features + label to data/landmarks_gnn.csv
- Press a key (0-9, a-b) to save, 'q' to quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import urllib.request

LANDMARKER_PATH = 'hand_landmarker.task'
LANDMARKER_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
CSV_FILE = 'data/landmarks_gnn.csv'
os.makedirs('data', exist_ok=True)

NUM_NODES = 42  # 21 per hand × 2 hands

SIGNS = {
    '0': 'Bird',
    '1': 'Boar',
    '2': 'Dog',
    '3': 'Dragon',
    '4': 'Hare',
    '5': 'Horse',
    '6': 'Monkey',
    '7': 'Ox',
    '8': 'Ram',
    '9': 'Rat',
    'a': 'Snake',
    'b': 'Tiger',
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


def download_model():
    if not os.path.exists(LANDMARKER_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("Download complete.")


def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts  # (21, 3)


def draw_landmarks(frame, landmarks, h, w, color=(0, 200, 0)):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)


def main():
    download_model()

    write_header = not os.path.exists(CSV_FILE)
    csv_file = open(CSV_FILE, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        # hand0: nodes 0-20, hand1: nodes 21-41
        header = [f'n{i}_{c}' for i in range(NUM_NODES) for c in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    counts = {k: 0 for k in SIGNS}

    print("Press 0-9 or a/b to save a sample (both hands). Press 'q' to quit.\n")
    for k, v in SIGNS.items():
        print(f"  [{k}] {v}")

    colors = [(0, 200, 0), (0, 100, 255)]  # green = hand0, orange = hand1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            num_detected = len(result.hand_landmarks) if result.hand_landmarks else 0

            # Build 42-node feature: hand0 (21 nodes) + hand1 (21 nodes, zeros if missing)
            hand0 = np.zeros((21, 3), dtype=np.float32)
            hand1 = np.zeros((21, 3), dtype=np.float32)

            if num_detected >= 1:
                hand0 = normalize_landmarks(result.hand_landmarks[0])
                draw_landmarks(frame, result.hand_landmarks[0], h, w, colors[0])
            if num_detected >= 2:
                hand1 = normalize_landmarks(result.hand_landmarks[1])
                draw_landmarks(frame, result.hand_landmarks[1], h, w, colors[1])

            node_features = np.concatenate([hand0, hand1], axis=0)  # (42, 3)
            detected = num_detected >= 1

            # HUD
            y = 30
            for k, v in SIGNS.items():
                cv2.putText(frame, f'[{k}] {v}: {counts[k]}', (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if detected else (100, 100, 100), 1)
                y += 20

            hands_str = f"{num_detected}/2 hand(s) detected"
            cv2.putText(frame, hands_str, (10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if num_detected == 2 else (0, 165, 255) if num_detected == 1 else (0, 0, 255), 2)

            cv2.imshow('Landmark Collection (GNN)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            ch = chr(key) if key < 128 else ''
            if ch in SIGNS and detected:
                label = SIGNS[ch]
                row = node_features.flatten().tolist() + [label]
                writer.writerow(row)
                csv_file.flush()
                counts[ch] += 1
                print(f"Saved [{ch}] {label} ({num_detected} hand(s)) — total: {counts[ch]}")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nDone. Saved to {CSV_FILE}")
    for k, v in SIGNS.items():
        print(f"  {v}: {counts[k]}")


if __name__ == '__main__':
    main()
