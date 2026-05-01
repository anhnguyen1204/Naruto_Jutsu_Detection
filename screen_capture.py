"""
Screen capture tool — shows a draggable 400x400 transparent overlay.
Position the box over the hand sign, then press:
  - 0-9 / a-b  to capture and save to the correct class folder
  - 'q'        to quit
"""

import tkinter as tk
import os
from PIL import ImageGrab

CROP_SIZE = 500
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

for sign in SIGNS.values():
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)


def get_next_index(label):
    folder = os.path.join(DATA_DIR, label)
    nums = []
    for f in os.listdir(folder):
        if f.startswith(label) and f.endswith('.jpg'):
            try:
                nums.append(int(f[len(label):-4]))
            except ValueError:
                pass
    return max(nums, default=0)


def save_capture(root, canvas, x, y):
    # x, y are the window's top-left screen position
    left = x
    top = y
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE

    # Hide window briefly so it doesn't appear in the screenshot
    root.withdraw()
    root.update()

    img = ImageGrab.grab(bbox=(left, top, right, bottom))

    root.deiconify()
    return img


class OverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Capture")
        self.root.geometry(f"{CROP_SIZE}x{CROP_SIZE + 60}+400+200")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.4)
        self.root.configure(bg='black')
        self.root.resizable(False, False)

        # Drag support
        self._drag_x = 0
        self._drag_y = 0
        self.root.bind('<ButtonPress-1>', self.start_drag)
        self.root.bind('<B1-Motion>', self.do_drag)

        # Key bindings
        for key in list(SIGNS.keys()) + ['q']:
            self.root.bind(f'<Key-{key}>', self.on_key)
        self.root.bind('<FocusIn>', lambda e: self.update_label())

        # Capture zone indicator
        self.canvas = tk.Canvas(root, width=CROP_SIZE, height=CROP_SIZE,
                                bg='black', highlightthickness=2,
                                highlightbackground='orange')
        self.canvas.pack()
        self.canvas.create_text(CROP_SIZE // 2, CROP_SIZE // 2,
                                text="Position over hand sign\nthen press key to capture",
                                fill='white', font=('Arial', 12), justify='center')

        # Status bar
        self.status = tk.Label(root, text="Keys: 0-9, a, b  |  q=quit",
                               bg='black', fg='white', font=('Arial', 9))
        self.status.pack(fill='x')

        self.counts = tk.Label(root, text=self._counts_text(),
                               bg='black', fg='#aaffaa', font=('Arial', 8))
        self.counts.pack(fill='x')

        self.update_label()

    def _counts_text(self):
        parts = [f"{v}:{get_next_index(v)}" for v in SIGNS.values()]
        return '  '.join(parts)

    def update_label(self):
        self.counts.config(text=self._counts_text())

    def start_drag(self, e):
        self._drag_x = e.x
        self._drag_y = e.y

    def do_drag(self, e):
        x = self.root.winfo_x() + (e.x - self._drag_x)
        y = self.root.winfo_y() + (e.y - self._drag_y)
        self.root.geometry(f'+{x}+{y}')

    def on_key(self, e):
        ch = e.char
        if ch == 'q':
            self.root.destroy()
            return
        if ch not in SIGNS:
            return

        label = SIGNS[ch]
        win_x = self.root.winfo_x()
        win_y = self.root.winfo_y()

        img = save_capture(self.root, self.canvas, win_x, win_y)

        next_idx = get_next_index(label) + 1
        filename = os.path.join(DATA_DIR, label, f'{label}{next_idx}.jpg')
        img.save(filename)
        print(f"Saved → {filename}")
        self.update_label()
        self.status.config(text=f"Saved {label}{next_idx}.jpg  |  q=quit")


def main():
    root = tk.Tk()
    app = OverlayApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
