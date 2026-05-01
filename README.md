# Naruto Hand Sign Recognition

Real-time Naruto jutsu hand sign recognition using CNN and GNN, with live webcam inference and jutsu sequence detection.

---

## Project Structure

```
NarutoHandSign/
├── collect_data.py          # Step 1: Collect hand sign images via webcam (CNN dataset)
├── collect_landmarks.py     # Step 2: Collect hand landmarks via MediaPipe (GNN dataset)
├── screen_capture.py        # Optional: Capture screen region for the dataset
├── train.ipynb              # Step 3: Train MobileNetV2  (CNN)
├── train_gnn.ipynb          # Step 4: Train Graph Attention Network (GNN)
├── inference.py             # Step 5: Real-time inference (choose CNN or GNN at startup)
├── data/
│   ├── images/              # CNN dataset — one folder per class
│   └── landmarks_gnn.csv    # GNN dataset — 42 landmarks per sample
├── model/
│   ├── mobilenet_v2.pt      # Trained MobileNetV2 weights
│   ├── gnn.pt               # Trained GAT weights
│   ├── label_map.json       # CNN class index → label
│   ├── label_map_gnn.json   # GNN class index → label
├── hand_landmarker.task     # MediaPipe hand landmark model (auto-downloaded)
└── requirements.txt
```

---

## Hand Signs (12 Classes)

| Key | Sign   | Key | Sign   |
|-----|--------|-----|--------|
| 0   | Bird   | 6   | Monkey |
| 1   | Boar   | 7   | Ox     |
| 2   | Dog    | 8   | Ram    |
| 3   | Dragon | 9   | Rat    |
| 4   | Hare   | a   | Snake  |
| 5   | Horse  | b   | Tiger  |

---

## Jutsu Sequences

| Jutsu        | Sequence              |
|--------------|-----------------------|
| Shadow Clone | Ram → Snake → Tiger   |
| Fireball     | Horse → Tiger → Dog   |
| Chidori      | Ox → Hare → Monkey    |
| Rasengan     | Bird → Dragon → Rat   |
| Rock Fist    | Boar → Ram → Snake    |

---

## Setup

```bash
pip install -r requirements.txt
pip install torch-geometric
```

> `hand_landmarker.task` is downloaded automatically on first run of `inference.py`.

---

## Workflow

### Step 1 — Collect CNN Images
```bash
python collect_data.py
```
- Opens webcam with a fixed 500x500 center crop box
- Press `0–9`, `a`, `b` to save a sample for that class
- Press `q` to quit

### Step 2 — Collect GNN Landmarks
```bash
python collect_landmarks.py
```
- Uses MediaPipe to detect up to 2 hands
- Saves 42 normalized landmarks (x, y, z) per sample to `data/landmarks_gnn.csv`

### Step 3 — Train CNN
Open and run `train.ipynb`:
- Trains **MobileNetV2** with transfer learning
- 80% of backbone frozen, custom classifier head
- Early stopping (patience = 8), cosine LR schedule
- Plots train vs val accuracy for both models

### Step 4 — Train GNN
Open and run `train_gnn.ipynb`:
- Trains a **Graph Attention Network (GAT)** on hand landmark graphs
- 42 nodes, 86 bidirectional edges, 4 GATBlocks with residual connections + BatchNorm
- Early stopping (patience = 10), AdamW optimizer, label smoothing
- Plots train vs val accuracy curve

### Step 5 — Run Inference
```bash
python inference.py
```
- Choose mode at startup:
  - `1` — CNN (MobileNetV2, crops 500×500 region centered on detected hand)
  - `2` — GNN (GAT, uses 42-node hand landmark graph)
- Hold a sign steady for **15 frames** at confidence ≥ 0.75 to confirm it
- Confirmed signs build up in a rolling buffer → jutsu name displayed on match
- Press `q` to quit

---

## Models & Results

| Model         | Input              | Trainable Params | Val Accuracy | Stopped  |
|---------------|--------------------|------------------|--------------|----------|
| MobileNetV2   | 500×500 image crop | ~1.8M            | 98.4%        | Epoch 13 |
| GAT (4-layer) | 42-node hand graph | ~300K            | 97.9%        | Epoch 44 |

---

## CNN vs GNN

| Aspect          | CNN                           | GNN                                 |
|-----------------|-------------------------------|-------------------------------------|
| Input           | Raw pixel image crop          | Hand landmark coordinates           |
| Strength        | Captures texture & appearance | Invariant to lighting & background  |
| Weakness        | Sensitive to lighting/angle   | Requires accurate hand detection    |
| Dataset size    | 5354 images   | 1,040 landmark samples              |

---

## Optional Tools

**Capture screen region:**
```bash
python screen_capture.py
```
Captures a selected screen region and saves it as training data.
