"""
inference.py — Real-time Naruto jutsu detection with visual effects.

Modes (choose at startup):
  1 — CNN  (MobileNetV2 on cropped hand image)
  2 — GNN  (GAT on 42-node landmark graph, both hands)

Hold-to-confirm: sign must be stable for HOLD_FRAMES at >= CONFIDENCE_THRESHOLD
Confirmed signs build a rolling buffer -> jutsu detected on sequence match.
On jutsu activation: particle effects rendered and tracked on the detected hand.
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
import math
import random
import urllib.request
from collections import deque
from typing import Optional, List

# ── File paths ────────────────────────────────────────────────────────────────
CNN_MODEL_FILE  = 'model/mobilenet_v2.pt'
CNN_LABEL_MAP   = 'model/label_map.json'
GNN_MODEL_FILE  = 'model/gnn.pt'
GNN_LABEL_MAP   = 'model/label_map_gnn.json'
LANDMARKER_PATH = 'hand_landmarker.task'
LANDMARKER_URL  = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task'
)

# ── Inference hyperparameters ─────────────────────────────────────────────────
IMAGE_SIZE           = 224
CROP_SIZE            = 500
CONFIDENCE_THRESHOLD = 0.75
HOLD_FRAMES          = 8
BUFFER_SIZE          = 10

# ── Jutsu sequences ───────────────────────────────────────────────────────────
JUTSU_SEQUENCES = {
    'Shadow Clone': ['Ram',   'Snake',  'Tiger'],
    'Fireball':     ['Horse', 'Tiger',  'Dog'],
    'Chidori':      ['Ox',    'Hare',   'Monkey'],
    'Rasengan':     ['Bird',  'Dragon', 'Rat'],
    'Rock Fist':    ['Boar',  'Ram',    'Snake'],
}
JUTSU_TUPLES = {k: tuple(v) for k, v in JUTSU_SEQUENCES.items()}

# ── Hand graph edges (2 hands, 42 nodes) ─────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]
_EDGES  = HAND_CONNECTIONS + [(a+21, b+21) for a,b in HAND_CONNECTIONS] + [(0,21)]
_src    = [a for a,b in _EDGES] + [b for a,b in _EDGES]
_dst    = [b for a,b in _EDGES] + [a for a,b in _EDGES]
EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)

# ── CNN preprocessing ─────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =============================================================================
#  PARTICLE + EFFECT SYSTEM
# =============================================================================

class Particle:
    __slots__ = ['x','y','vx','vy','life','age','color','size']

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.life        = life
        self.age         = 0
        self.color       = color
        self.size        = size

    def step(self, gravity=0.0):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += gravity
        self.age += 1

    @property
    def alive(self):
        return self.age < self.life

    @property
    def alpha(self):
        return max(0.0, 1.0 - self.age / self.life)


class JutsuEffect:
    """Visual effect that spawns on jutsu activation and tracks the hand."""

    DURATION = 210  # ~7 s at 30 fps

    def __init__(self, name, cx, cy):
        self.name      = name
        self.cx        = float(cx)
        self.cy        = float(cy)
        self.frame     = 0
        self.particles = []

    # -- particle factories ---------------------------------------------------

    def _fire_particle(self):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(1.5, 6.5)
        color = random.choice([
            (0, 255, 255),
            (0, 210, 255),
            (0, 140, 255),
            (0,  70, 240),
            (0,  30, 200),
        ])
        return Particle(
            self.cx + random.uniform(-12, 12),
            self.cy + random.uniform(-12, 12),
            math.cos(angle)*speed,
            math.sin(angle)*speed - 1.2,
            random.randint(14, 38), color, random.randint(3, 11),
        )

    def _rock_particle(self):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(0.5, 4.5)
        g     = random.randint(65, 145)
        return Particle(
            self.cx + random.uniform(-45, 45),
            self.cy + random.uniform(-45, 45),
            math.cos(angle)*speed, math.sin(angle)*speed,
            random.randint(22, 55), (g-15, g-25, g+10), random.randint(6, 17),
        )

    def _rasengan_particle(self):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(2.5, 5.5)
        return Particle(
            self.cx, self.cy,
            math.cos(angle)*speed, math.sin(angle)*speed,
            random.randint(10, 22), (255, 230, 180), random.randint(2, 6),
        )

    def _chidori_particle(self):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(3.0, 8.0)
        return Particle(
            self.cx + random.uniform(-8, 8),
            self.cy + random.uniform(-8, 8),
            math.cos(angle)*speed, math.sin(angle)*speed,
            random.randint(6, 18), (255, 245, 200), random.randint(2, 5),
        )

    # -- update ---------------------------------------------------------------

    def update(self, cx, cy):
        self.cx = float(cx)
        self.cy = float(cy)
        self.frame += 1

        if self.name == 'Fireball':
            for _ in range(6):
                self.particles.append(self._fire_particle())
        elif self.name == 'Rock Fist' and self.frame < 90:
            for _ in range(2):
                self.particles.append(self._rock_particle())
        elif self.name == 'Rasengan':
            for _ in range(4):
                self.particles.append(self._rasengan_particle())
        elif self.name == 'Chidori' and self.frame % 3 == 0:
            for _ in range(5):
                self.particles.append(self._chidori_particle())

        gravity = {'Fireball': -0.14, 'Rock Fist': 0.22}.get(self.name, 0.0)
        for p in self.particles:
            p.step(gravity)
        self.particles = [p for p in self.particles if p.alive]

    # -- render ---------------------------------------------------------------

    def render(self, img):
        cx, cy = int(self.cx), int(self.cy)
        t      = self.frame
        if   self.name == 'Rasengan':     self._draw_rasengan(img, cx, cy, t)
        elif self.name == 'Fireball':     self._draw_fireball(img, cx, cy, t)
        elif self.name == 'Chidori':      self._draw_chidori(img, cx, cy, t)
        elif self.name == 'Shadow Clone': self._draw_shadow_clone(img, cx, cy, t)
        elif self.name == 'Rock Fist':    self._draw_rock_fist(img, cx, cy, t)

    @staticmethod
    def _blend(img, overlay, alpha):
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    def _draw_rasengan(self, img, cx, cy, t):
        ov    = img.copy()
        pulse = int(7 * math.sin(t * 0.22))
        for r, c in [
            (100+pulse, (190,  55,  5)),
            ( 84+pulse, (225, 115, 35)),
            ( 68,       (245, 175, 75)),
            ( 52,       (255, 225, 130)),
            ( 36,       (255, 250, 205)),
            ( 18,       (255, 255, 255)),
        ]:
            cv2.circle(ov, (cx, cy), max(1, r), c, -1)
        for i in range(12):
            a = t*0.10 + i*2*math.pi/12
            cv2.circle(ov, (int(cx+94*math.cos(a)), int(cy+94*math.sin(a))), 5, (255, 248, 185), -1)
        for i in range(8):
            a = -t*0.16 + i*2*math.pi/8
            cv2.circle(ov, (int(cx+50*math.cos(a)), int(cy+50*math.sin(a))), 8, (210, 235, 255), -1)
        for i in range(6):
            a  = t*0.20 + i*math.pi/3
            cv2.line(ov,
                     (int(cx+22*math.cos(a)), int(cy+22*math.sin(a))),
                     (int(cx+64*math.cos(a)), int(cy+64*math.sin(a))),
                     (185, 225, 255), 2)
        self._blend(img, ov, 0.82)
        for p in self.particles:
            a  = p.alpha
            cv2.circle(img, (int(p.x), int(p.y)),
                       max(1, int(p.size*a)),
                       tuple(int(ch*a) for ch in p.color), -1)

    def _draw_fireball(self, img, cx, cy, t):
        ov        = img.copy()
        intensity = min(1.0, t / 22.0)
        base_r    = int((38 + 20*math.sin(t*0.22)) * intensity)
        if base_r > 0:
            for r, c in [
                (base_r+22, (0,  25, 150)),
                (base_r+13, (0,  85, 225)),
                (base_r,    (0, 165, 255)),
                (base_r- 9, (0, 225, 255)),
                (max(1, base_r-19), (60, 248, 255)),
            ]:
                cv2.circle(ov, (cx, cy), max(1, r), c, -1)
        self._blend(img, ov, 0.58)
        for p in self.particles:
            cv2.circle(img, (int(p.x), int(p.y)),
                       max(1, int(p.size*p.alpha)), p.color, -1)

    def _draw_chidori(self, img, cx, cy, t):
        ov = img.copy()
        for r, c in [
            (58, (210, 205, 255)),
            (42, (230, 235, 255)),
            (26, (248, 252, 255)),
            (12, (255, 255, 255)),
        ]:
            cv2.circle(ov, (cx, cy), r, c, -1)
        rng = random.Random(t)
        for _ in range(14):
            angle  = rng.uniform(0, 2*math.pi)
            length = rng.uniform(65, 190)
            prev   = (cx, cy)
            segs   = rng.randint(5, 10)
            for seg in range(segs):
                frac  = (seg + 1) / segs
                nx = int(cx + length*frac*math.cos(angle) + rng.uniform(-28, 28))
                ny = int(cy + length*frac*math.sin(angle) + rng.uniform(-28, 28))
                thick  = max(1, 3 - seg//3)
                bright = rng.randint(195, 255)
                cv2.line(ov, prev, (nx, ny), (255, bright, bright), thick)
                prev = (nx, ny)
        self._blend(img, ov, 0.82)
        flash = img.copy()
        cv2.circle(flash, (cx, cy), 14 + rng.randint(0, 10), (255, 255, 255), -1)
        self._blend(img, flash, 0.55)
        for p in self.particles:
            cv2.circle(img, (int(p.x), int(p.y)),
                       max(1, int(p.size*p.alpha)), p.color, -1)

    def _draw_shadow_clone(self, img, cx, cy, t):
        h, w  = img.shape[:2]
        drift = int(50 * math.sin(t * 0.035)) + 130
        for dx, dy in [(-drift, -8), (drift, -8)]:
            M       = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(img, M, (w, h))
            ghost   = shifted.copy()
            ghost[:, :, 2] = (ghost[:, :, 2] * 0.15).astype(np.uint8)
            ghost[:, :, 1] = (ghost[:, :, 1] * 0.35).astype(np.uint8)
            cv2.addWeighted(ghost, 0.28, img, 0.72, 0, img)
        ov = img.copy()
        cv2.circle(ov, (cx, cy), 55, (220, 70, 20), -1)
        self._blend(img, ov, 0.18)
        ring_r = int(60 + 30 * math.sin(t * 0.18))
        cv2.circle(img, (cx, cy), ring_r,    (200, 60, 15), 2)
        cv2.circle(img, (cx, cy), ring_r+18, (150, 40, 10), 1)

    def _draw_rock_fist(self, img, cx, cy, t):
        ov       = img.copy()
        progress = min(1.0, t / 20.0)
        r        = int(68 * progress)
        if r > 2:
            for rad, c in [
                (r+16, (48, 42, 58)),
                (r+ 8, (68, 58, 72)),
                (r,    (88, 78, 88)),
                (r-10, (108, 98, 105)),
            ]:
                cv2.circle(ov, (cx, cy), max(1, rad), c, -1)
            rng_f = random.Random(42)
            for _ in range(12):
                a  = rng_f.uniform(0, 2*math.pi)
                l  = rng_f.uniform(r*0.35, r*1.15)
                cv2.line(ov, (cx, cy),
                         (int(cx+l*math.cos(a)), int(cy+l*math.sin(a))),
                         (32, 28, 38), 2)
        self._blend(img, ov, 0.68)
        for p in self.particles:
            a  = p.alpha
            px, py = int(p.x), int(p.y)
            sz = max(2, int(p.size*a))
            pts = np.array([
                [px-sz,    py-sz//2],
                [px,       py-sz  ],
                [px+sz,    py-sz//2],
                [px+sz//2, py+sz//2],
                [px-sz//2, py+sz//2],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], p.color)

    def is_done(self):
        return self.frame >= self.DURATION


# =============================================================================
#  SIGN CONFIRMATION POPUP
# =============================================================================

class SignPopup:
    """Large text that floats upward and fades when a sign is confirmed."""
    LIFETIME = 45

    def __init__(self, sign_name, cx, cy):
        self.text  = sign_name.upper()
        self.cx    = cx
        self.cy    = cy
        self.frame = 0

    def render(self, img):
        alpha = max(0.0, 1.0 - self.frame / self.LIFETIME)
        y_off = int(-45 * (self.frame / self.LIFETIME))
        cy    = self.cy + y_off
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.15
        thick = 2
        ts    = cv2.getTextSize(self.text, font, scale, thick)[0]
        tx    = self.cx - ts[0] // 2
        ov = img.copy()
        cv2.putText(ov, self.text, (tx+2, cy+2), font, scale, (0,0,0), thick+2)
        cv2.putText(ov, self.text, (tx,   cy  ), font, scale, (0, 240, 255), thick)
        cv2.addWeighted(ov, alpha, img, 1.0-alpha, 0, img)
        self.frame += 1

    def is_done(self):
        return self.frame >= self.LIFETIME


# =============================================================================
#  HUD HELPERS
# =============================================================================

JUTSU_COLORS = {
    'Rasengan':     (230, 160,  60),
    'Fireball':     (  0, 100, 255),
    'Chidori':      (255, 255, 200),
    'Shadow Clone': (210,  80,  40),
    'Rock Fist':    ( 75,  75, 155),
}

JUTSU_SEQ_LABEL = {k: ' > '.join(v) for k, v in JUTSU_SEQUENCES.items()}


def draw_sequence_bar(img, sign_buffer, h, w):
    if not sign_buffer:
        return
    signs  = list(sign_buffer)
    n      = len(signs)
    bar_y  = 52
    bar_h  = 46
    slot_w = min(155, (w - 24) // max(1, n))
    total  = slot_w * n
    sx     = (w - total) // 2
    ov = img.copy()
    cv2.rectangle(ov, (sx-6, bar_y), (sx+total+6, bar_y+bar_h), (15,15,15), -1)
    cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)
    for i, sign in enumerate(signs):
        x  = sx + i * slot_w
        cv2.rectangle(img, (x, bar_y+2), (x+slot_w-6, bar_y+bar_h-2), (0, 200, 200), 1)
        ts = cv2.getTextSize(sign, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        tx = x + (slot_w - 6 - ts[0]) // 2
        cv2.putText(img, sign, (tx, bar_y+bar_h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
        if i < n - 1:
            ax = x + slot_w - 3
            cv2.arrowedLine(img, (ax, bar_y+bar_h//2),
                            (ax+6, bar_y+bar_h//2), (180,180,180), 1, tipLength=0.6)


def draw_jutsu_sequences_hint(img, w):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    lh    = 18
    pad   = 8
    lines = list(JUTSU_SEQ_LABEL.items())
    max_w = max(cv2.getTextSize(f"{n}: {s}", font, scale, thick)[0][0]
                for n, s in lines) + 10
    bx    = w - max_w - pad - 5
    by    = 25
    ov = img.copy()
    cv2.rectangle(ov, (bx-pad, by-lh), (w-5, by+lh*len(lines)+pad), (12,12,12), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for i, (name, seq) in enumerate(lines):
        color = JUTSU_COLORS.get(name, (200, 200, 200))
        cv2.putText(img, f"{name}: {seq}", (bx, by+i*lh), font, scale, color, thick)


def draw_jutsu_banner(img, jutsu_name, timer, h, w):
    TOTAL = 90
    if timer > TOTAL - 15:
        alpha = (TOTAL - timer) / 15.0
    elif timer < 15:
        alpha = timer / 15.0
    else:
        alpha = 1.0
    alpha = max(0.0, min(1.0, alpha))

    color = JUTSU_COLORS.get(jutsu_name, (0, 255, 255))
    text  = jutsu_name.upper()
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 2.0
    thick = 3
    ts = cv2.getTextSize(text, font, scale, thick)[0]
    tx = (w - ts[0]) // 2
    ty = h // 2 + 18

    ov = img.copy()
    cv2.rectangle(ov, (0, ty-60), (w, ty+25), (0,0,0), -1)
    cv2.line(ov, (0, ty-62), (w, ty-62), color, 2)
    cv2.line(ov, (0, ty+27), (w, ty+27), color, 2)
    cv2.putText(ov, text, (tx+3, ty+3), font, scale, (0,0,0), thick+3)
    cv2.putText(ov, text, (tx,   ty  ), font, scale, color,   thick)
    sub   = "-- JUTSU ACTIVATED --"
    sub_s = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(ov, sub, ((w-sub_s[0])//2, ty-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.addWeighted(ov, alpha, img, 1.0-alpha, 0, img)


def draw_hold_indicator(img, candidate, hold_count, h):
    if not candidate or hold_count <= 0:
        return
    prog = int((hold_count / HOLD_FRAMES) * 220)
    cv2.rectangle(img, (10, h-11), (10+prog, h-3), (0, 165, 255), -1)
    cv2.putText(img, candidate, (12, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


# =============================================================================
#  MODEL DEFINITIONS
# =============================================================================

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


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def download_landmarker():
    if not os.path.exists(LANDMARKER_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("Download complete.")


def normalize_landmarks(landmarks):
    pts   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts  -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts  /= scale
    return pts


def draw_landmarks_on_frame(frame, landmarks, h, w, color=(0, 200, 0)):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)


def get_hand_center(landmarks, h, w):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def check_jutsu(buffer):
    buf = tuple(buffer)
    for name, seq in JUTSU_TUPLES.items():
        n = len(seq)
        if len(buf) >= n and buf[-n:] == seq:
            return name
    return None


def choose_mode():
    print("\nSelect inference mode:")
    print("  1 - CNN (MobileNetV2 on hand crop image)")
    print("  2 - GNN (GAT on 42-node landmark graph, both hands)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            return choice
        print("Invalid choice, enter 1 or 2.")


# =============================================================================
#  MAIN LOOP
# =============================================================================

def main():
    download_landmarker()
    mode = choose_mode()

    if mode == '1':
        with open(CNN_LABEL_MAP) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        model = build_cnn(len(label_map))
        model.load_state_dict(torch.load(CNN_MODEL_FILE, map_location='cpu'))
        model.eval()
        mode_name = "CNN - MobileNetV2"
        print("CNN loaded. Press 'q' to quit.")
    else:
        with open(GNN_LABEL_MAP) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        model = GAT(in_channels=3, hidden_channels=128,
                    num_classes=len(label_map), num_layers=4)
        model.load_state_dict(torch.load(GNN_MODEL_FILE, map_location='cpu'))
        model.eval()
        mode_name = "GNN - GAT (2 hands)"
        print("GNN loaded. Press 'q' to quit.")

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

    # State
    sign_buffer       = deque(maxlen=BUFFER_SIZE)
    current_candidate = None
    hold_count        = 0
    last_confirmed    = None

    jutsu_display     = None
    jutsu_timer       = 0

    active_effect     = None
    popups            = []

    last_hand_cx, last_hand_cy = 320, 240
    hand_colors = [(0, 200, 0), (0, 100, 255)]

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 1
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            pred_sign  = None
            confidence = 0.0
            hand_cx, hand_cy = last_hand_cx, last_hand_cy

            # -- CNN inference ------------------------------------------------
            if mode == '1' and result.hand_landmarks:
                lms = result.hand_landmarks[0]
                xs  = [lm.x * w for lm in lms]
                ys  = [lm.y * h for lm in lms]
                cx  = int((min(xs) + max(xs)) / 2)
                cy  = int((min(ys) + max(ys)) / 2)
                hand_cx, hand_cy = cx, cy
                half = CROP_SIZE // 2
                x1 = max(0, cx - half);      y1 = max(0, cy - half)
                x2 = min(w, x1 + CROP_SIZE); y2 = min(h, y1 + CROP_SIZE)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                crop_rgb = cv2.cvtColor(
                    cv2.resize(frame[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE)),
                    cv2.COLOR_BGR2RGB)
                tensor = preprocess(crop_rgb).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]
                conf, idx = probs.max(0)
                confidence = conf.item()
                if confidence >= CONFIDENCE_THRESHOLD:
                    pred_sign = label_map[idx.item()]
                preview = cv2.resize(frame[y1:y2, x1:x2], (112, 112))
                frame[10:122, w-122:w-10] = preview

            # -- GNN inference ------------------------------------------------
            elif mode == '2' and result.hand_landmarks:
                for i, lms in enumerate(result.hand_landmarks):
                    draw_landmarks_on_frame(frame, lms, h, w, hand_colors[i % 2])
                hand_cx, hand_cy = get_hand_center(result.hand_landmarks[0], h, w)
                hand0 = normalize_landmarks(result.hand_landmarks[0])
                hand1 = (normalize_landmarks(result.hand_landmarks[1])
                         if len(result.hand_landmarks) >= 2
                         else np.zeros((21, 3), dtype=np.float32))
                pts_np = np.concatenate([hand0, hand1], axis=0)
                x_t    = torch.tensor(pts_np, dtype=torch.float)
                b_t    = torch.zeros(42, dtype=torch.long)
                with torch.no_grad():
                    probs = torch.softmax(model(x_t, EDGE_INDEX, b_t), dim=1)[0]
                conf, idx = probs.max(0)
                confidence = conf.item()
                if confidence >= CONFIDENCE_THRESHOLD:
                    pred_sign = label_map[idx.item()]

            last_hand_cx, last_hand_cy = hand_cx, hand_cy

            # -- Hold-to-confirm ----------------------------------------------
            if pred_sign is not None:
                if pred_sign == current_candidate:
                    hold_count += 1
                else:
                    current_candidate = pred_sign
                    hold_count = 1

                if hold_count >= HOLD_FRAMES and pred_sign != last_confirmed:
                    sign_buffer.append(pred_sign)
                    last_confirmed = pred_sign
                    hold_count     = 0
                    print(f"  Confirmed: {pred_sign}  |  buffer: {list(sign_buffer)}")

                    popups.append(SignPopup(pred_sign, hand_cx, hand_cy - 70))

                    jutsu = check_jutsu(sign_buffer)
                    if jutsu:
                        jutsu_display = jutsu
                        jutsu_timer   = 90
                        active_effect = JutsuEffect(jutsu, hand_cx, hand_cy)
                        sign_buffer.clear()
                        last_confirmed = None
                        print(f"\n*** JUTSU: {jutsu} ***\n")
            else:
                current_candidate = None
                hold_count        = 0

            # -- Update & render visual effect --------------------------------
            if active_effect is not None:
                active_effect.update(hand_cx, hand_cy)
                active_effect.render(frame)
                if active_effect.is_done():
                    active_effect = None

            # -- Sign popups --------------------------------------------------
            for popup in popups:
                popup.render(frame)
            popups = [p for p in popups if not p.is_done()]

            # -- Sequence buffer bar ------------------------------------------
            draw_sequence_bar(frame, sign_buffer, h, w)

            # -- Jutsu hint table ---------------------------------------------
            draw_jutsu_sequences_hint(frame, w)

            # -- Jutsu activation banner --------------------------------------
            if jutsu_timer > 0:
                draw_jutsu_banner(frame, jutsu_display, jutsu_timer, h, w)
                jutsu_timer -= 1

            # -- Base HUD strip -----------------------------------------------
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 48), (10, 10, 10), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, mode_name, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            if pred_sign and active_effect is None:
                cv2.putText(frame, f'{pred_sign}  {confidence:.0%}',
                            (10, h-48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                bar_w = int(confidence * 220)
                cv2.rectangle(frame, (10, h-28), (10+bar_w, h-14), (0,200,0), -1)
                cv2.rectangle(frame, (10, h-28), (230,      h-14), (180,180,180), 1)

            draw_hold_indicator(frame, current_candidate, hold_count, h)

            cv2.imshow('Naruto Jutsu Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == '__main__':
    main()
