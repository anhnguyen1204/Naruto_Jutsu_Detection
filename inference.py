"""
Step 5 + 6: Real-time inference + jutsu sequence detector.

Modes (choose at startup):
  1 — CNN only  (MobileNetV2 on cropped hand image)
  2 — GNN only  (GAT on 42-node landmark graph, both hands)

Hold-to-confirm: sign must be stable for HOLD_FRAMES at >= CONFIDENCE_THRESHOLD
Buffers confirmed signs and matches against known jutsu sequences.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import json
import os
import urllib.request
from collections import deque

# ── File paths ────────────────────────────────────────────────────────────────
CNN_MODEL_FILE  = 'model/mobilenet_v2.pt'
CNN_LABEL_MAP   = 'model/label_map.json'
GNN_MODEL_FILE  = 'model/gnn.pt'
GNN_LABEL_MAP   = 'model/label_map_gnn.json'
LANDMARKER_PATH = 'hand_landmarker.task'
LANDMARKER_URL  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE           = 224
CROP_SIZE            = 500
CONFIDENCE_THRESHOLD = 0.75
HOLD_FRAMES          = 8
BUFFER_SIZE          = 10

# ── Jutsu sequences ───────────────────────────────────────────────────────────
JUTSU_SEQUENCES = {
    'Shadow Clone': ['Ram', 'Snake', 'Tiger'],
    'Fireball':     ['Horse', 'Tiger', 'Dog'],
    'Chidori':      ['Ox', 'Hare', 'Monkey'],
    'Rasengan':     ['Bird', 'Dragon', 'Rat'],
    'Rock Fist':    ['Boar', 'Ram', 'Snake'],
}
JUTSU_TUPLES = {k: tuple(v) for k, v in JUTSU_SEQUENCES.items()}

# ── Hand graph edges — 2 hands (42 nodes) ────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]
_EDGES = HAND_CONNECTIONS + [(a+21, b+21) for a,b in HAND_CONNECTIONS] + [(0, 21)]
_src = [a for a,b in _EDGES] + [b for a,b in _EDGES]
_dst = [b for a,b in _EDGES] + [a for a,b in _EDGES]
EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)

# ── CNN preprocessing ─────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model definitions ─────────────────────────────────────────────────────────

def build_cnn(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


class GATBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads=4):
        super().__init__()
        self.gat  = GATConv(in_ch, out_ch // heads, heads=heads, dropout=0.3)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index):
        return torch.relu(self.bn(self.gat(x, edge_index)) + self.proj(x))


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            GATBlock(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x, edge_index)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_landmarker():
    if not os.path.exists(LANDMARKER_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("Download complete.")


def get_hand_bbox(all_landmarks, h, w):
    xs = [lm.x * w for lms in all_landmarks for lm in lms]
    ys = [lm.y * h for lms in all_landmarks for lm in lms]
    cx = int((min(xs) + max(xs)) / 2)
    cy = int((min(ys) + max(ys)) / 2)
    half = CROP_SIZE // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + CROP_SIZE)
    y2 = min(h, y1 + CROP_SIZE)
    return x1, y1, x2, y2


def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts


def draw_landmarks(frame, landmarks, h, w, color=(0, 200, 0)):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)


def check_jutsu(buffer):
    buf = tuple(buffer)
    for name, seq in JUTSU_TUPLES.items():
        n = len(seq)
        if len(buf) >= n and buf[-n:] == seq:
            return name
    return None


def choose_mode():
    print("\nSelect inference mode:")
    print("  1 — CNN (MobileNetV2 on hand crop image)")
    print("  2 — GNN (GAT on 42-node landmark graph, both hands)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            return choice
        print("Invalid choice, enter 1 or 2.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    download_landmarker()

    mode = choose_mode()

    if mode == '1':
        with open(CNN_LABEL_MAP) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        model = build_cnn(len(label_map))
        model.load_state_dict(torch.load(CNN_MODEL_FILE, map_location='cpu'))
        model.eval()
        print("CNN loaded. Press 'q' to quit.")
        mode_name = "CNN — MobileNetV2"
    else:
        with open(GNN_LABEL_MAP) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        model = GAT(in_channels=3, hidden_channels=128, num_classes=len(label_map), num_layers=4)
        model.load_state_dict(torch.load(GNN_MODEL_FILE, map_location='cpu'))
        model.eval()
        print("GNN loaded. Press 'q' to quit.")
        mode_name = "GNN — GAT (2 hands)"

    num_hands = 2 if mode == '2' else 1

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    sign_buffer      = deque(maxlen=BUFFER_SIZE)
    current_candidate = None
    hold_count       = 0
    last_confirmed   = None
    jutsu_display    = None
    jutsu_timer      = 0

    hand_colors = [(0, 200, 0), (0, 100, 255)]

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

            pred_sign  = None
            confidence = 0.0

            if mode == '1' and result.hand_landmarks:
                # ── CNN: crop centered on detected hand ──
                lms = result.hand_landmarks[0]
                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                cx = int((min(xs) + max(xs)) / 2)
                cy = int((min(ys) + max(ys)) / 2)
                half = CROP_SIZE // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, x1 + CROP_SIZE)
                y2 = min(h, y1 + CROP_SIZE)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                crop_bgr = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(
                    cv2.resize(crop_bgr, (IMAGE_SIZE, IMAGE_SIZE)),
                    cv2.COLOR_BGR2RGB
                )
                tensor = preprocess(crop_rgb).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]
                conf, idx = probs.max(dim=0)
                confidence = conf.item()
                if confidence >= CONFIDENCE_THRESHOLD:
                    pred_sign = label_map[idx.item()]

                preview = cv2.resize(crop_bgr, (112, 112))
                frame[10:122, w - 122:w - 10] = preview

            elif mode == '2' and result.hand_landmarks:
                for i, lms in enumerate(result.hand_landmarks):
                    draw_landmarks(frame, lms, h, w, hand_colors[i % 2])
                # ── GNN: 42 nodes (hand0 + hand1, zeros if only 1 hand) ──
                hand0 = normalize_landmarks(result.hand_landmarks[0])
                hand1 = normalize_landmarks(result.hand_landmarks[1]) \
                        if len(result.hand_landmarks) >= 2 \
                        else np.zeros((21, 3), dtype=np.float32)
                pts = np.concatenate([hand0, hand1], axis=0)
                x_tensor     = torch.tensor(pts, dtype=torch.float)
                batch_tensor = torch.zeros(42, dtype=torch.long)
                with torch.no_grad():
                    probs = torch.softmax(
                        model(x_tensor, EDGE_INDEX, batch_tensor), dim=1
                    )[0]
                conf, idx = probs.max(dim=0)
                confidence = conf.item()
                if confidence >= CONFIDENCE_THRESHOLD:
                    pred_sign = label_map[idx.item()]

            # ── Hold-to-confirm ───────────────────────────────────────────────
            if pred_sign is not None:
                if pred_sign == current_candidate:
                    hold_count += 1
                else:
                    current_candidate = pred_sign
                    hold_count = 1

                if hold_count >= HOLD_FRAMES and pred_sign != last_confirmed:
                    sign_buffer.append(pred_sign)
                    last_confirmed = pred_sign
                    hold_count = 0
                    print(f"Confirmed: {pred_sign}  |  Buffer: {list(sign_buffer)}")

                    jutsu = check_jutsu(sign_buffer)
                    if jutsu:
                        jutsu_display = jutsu
                        jutsu_timer = 90
                        print(f"*** JUTSU: {jutsu} ***")
                        sign_buffer.clear()
                        last_confirmed = None
            else:
                current_candidate = None
                hold_count = 0

            # ── HUD ───────────────────────────────────────────────────────────
            cv2.putText(frame, mode_name, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            if pred_sign:
                cv2.putText(frame, f'{pred_sign}  {confidence:.2f}', (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                bar_w = int(confidence * 200)
                cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 15), (0, 200, 0), -1)
                cv2.rectangle(frame, (10, h - 30), (210, h - 15), (200, 200, 200), 1)

            if current_candidate and hold_count > 0:
                prog = int((hold_count / HOLD_FRAMES) * 200)
                cv2.rectangle(frame, (10, h - 12), (10 + prog, h - 3), (255, 165, 0), -1)

            buf_str = ' → '.join(sign_buffer)
            cv2.putText(frame, buf_str, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)

            if jutsu_timer > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 2 - 50), (w, h // 2 + 50), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                cv2.putText(frame, jutsu_display, (w // 2 - 200, h // 2 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 200, 255), 3)
                jutsu_timer -= 1

            cv2.imshow('Naruto Hand Sign Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
