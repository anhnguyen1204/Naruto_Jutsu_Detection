"""
Step 2: Data Collection Script (CNN version)
- Opens webcam, draws a fixed 500x500 box in the center of the frame
- Press a key (0-9, a-c) to save the clipped box region for that class
- Press 'q' to quit
"""

import cv2
import os

CROP_SIZE = 400

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

DATA_DIR = 'data/images'


def main():
    for sign in SIGNS.values():
        os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    def next_index(label):
        folder = os.path.join(DATA_DIR, label)
        existing = [f for f in os.listdir(folder) if f.startswith(label) and f.endswith('.jpg')]
        nums = []
        for f in existing:
            try:
                nums.append(int(f[len(label):-4]))
            except ValueError:
                pass
        return max(nums, default=0)

    counts = {k: next_index(v) for k, v in SIGNS.items()}

    print("Press 0-9 or a/b/c to save a sample for each sign.")
    print("Press 'q' to quit.\n")
    for k, v in SIGNS.items():
        print(f"  [{k}] {v}  ({counts[k]} saved)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Fixed box centered in the frame
        cx, cy = w // 2, h // 2
        half = CROP_SIZE // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # HUD
        y = 30
        for k, v in SIGNS.items():
            cv2.putText(frame, f'[{k}] {v}: {counts[k]}', (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20

        cv2.imshow('Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        ch = chr(key) if key < 128 else ''
        if ch in SIGNS:
            label = SIGNS[ch]
            crop = frame[y1:y2, x1:x2]
            filename = os.path.join(DATA_DIR, label, f'{label}{counts[ch] + 1}.jpg')
            cv2.imwrite(filename, crop)
            counts[ch] += 1
            print(f"Saved [{ch}] {label} — total: {counts[ch]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")
    for k, v in SIGNS.items():
        print(f"  {v}: {counts[k]}")


if __name__ == '__main__':
    main()
