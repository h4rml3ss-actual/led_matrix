import tkinter as tk
from tkinter import filedialog
import tkinter.simpledialog as simpledialog
import json
from PIL import Image

GRID_WIDTH = 64
GRID_HEIGHT = 32
PIXEL_SIZE = 15
COLORS = ["black", "white", "red", "green", "blue", "purple", "orange", "pink"]
DEFAULT_COLOR = "black"

# Map color names to RGB for matching
COLOR_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203)
}

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

class FrameEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Protogen Frame Editor")
        self.canvas = tk.Canvas(master, width=GRID_WIDTH * PIXEL_SIZE, height=GRID_HEIGHT * PIXEL_SIZE)
        self.canvas.pack()
        self.selected_color = COLORS[1]
        self.grid = [[DEFAULT_COLOR for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.rects = {}
        self.undo_stack = []
        self.current_action = []

        self.build_palette()
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.is_drawing = False

        menu = tk.Menu(master)
        master.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save_frame)
        file_menu.add_command(label="Load", command=self.load_frame)
        file_menu.add_separator()
        file_menu.add_command(label="Import from Image", command=self.import_image)
        file_menu.add_command(label="Import from ASCII", command=self.import_ascii)
        file_menu.add_command(label="Export as ASCII", command=self.export_ascii)
        edit_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo)

    def build_palette(self):
        palette = tk.Frame(self.master)
        palette.pack()
        for color in COLORS:
            btn = tk.Button(
                palette,
                text=color[0].upper(),
                bg=color,
                width=4,
                command=lambda c=color: self.set_color(c)
            )
            btn.pack(side=tk.LEFT)
        btn = tk.Button(
            palette,
            text="E",
            width=6,
            command=lambda: self.set_color(DEFAULT_COLOR)
        )
        btn.pack(side=tk.LEFT)

    def set_color(self, color):
        self.selected_color = color

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = self.canvas.create_rectangle(
                    x * PIXEL_SIZE, y * PIXEL_SIZE,
                    (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
                    fill=DEFAULT_COLOR, outline="gray"
                )
                self.rects[(x, y)] = rect

    def on_click(self, event):
        self.is_drawing = True
        self.current_action = []
        self.paint(event)

    def on_drag(self, event):
        if self.is_drawing:
            self.paint(event)

    def on_release(self, event):
        self.is_drawing = False
        if self.current_action:
            self.undo_stack.append(self.current_action)

    def paint(self, event):
        x = event.x // PIXEL_SIZE
        y = event.y // PIXEL_SIZE
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            previous_color = self.grid[y][x]
            if previous_color != self.selected_color:
                self.current_action.append((x, y, previous_color))
                self.grid[y][x] = self.selected_color
                self.canvas.itemconfig(self.rects[(x, y)], fill=self.selected_color)

    def undo(self):
        if self.undo_stack:
            last_action = self.undo_stack.pop()
            for x, y, color in reversed(last_action):
                self.grid[y][x] = color
                self.canvas.itemconfig(self.rects[(x, y)], fill=color)

    def save_frame(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.grid, f)
            print(f"Saved to {file_path}")

    def load_frame(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "r") as f:
                self.grid = json.load(f)
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    color = self.grid[y][x]
                    self.canvas.itemconfig(self.rects[(x, y)], fill=color)
            print(f"Loaded from {file_path}")

    def import_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = Image.open(file_path).convert("RGB")
        img = img.resize((GRID_WIDTH, GRID_HEIGHT))
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pixel = img.getpixel((x, y))
                best_color = min(COLOR_RGB.items(), key=lambda c: color_distance(c[1], pixel))[0]
                self.grid[y][x] = best_color
                self.canvas.itemconfig(self.rects[(x, y)], fill=best_color)

    def import_ascii(self):
        ascii_text = simpledialog.askstring("Paste ASCII Frame", "Paste frame ASCII below:")
        if not ascii_text:
            return
        lines = ascii_text.strip().split("\n")
        symbol_to_color = {
            ".": "black", "W": "white", "R": "red", "G": "green", "B": "blue",
            "P": "purple", "O": "orange", "K": "pink"
        }
        for y, line in enumerate(lines):
            tokens = line.strip().split()
            for x, symbol in enumerate(tokens):
                color = symbol_to_color.get(symbol.upper(), "black")
                if x < GRID_WIDTH and y < GRID_HEIGHT:
                    self.grid[y][x] = color
                    self.canvas.itemconfig(self.rects[(x, y)], fill=color)

    def export_ascii(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if not file_path:
            return
        color_to_symbol = {
            "black": ".", "white": "W", "red": "R", "green": "G",
            "blue": "B", "purple": "P", "orange": "O", "pink": "K"
        }

        with open(file_path, "w") as f:
            # Write column headers
            col_header = "     " + " ".join(f"{i:2}" for i in range(GRID_WIDTH))
            f.write(col_header + "\n")
            f.write("     +" + "-" * (3 * GRID_WIDTH - 1) + "\n")

            # Write each row with index and separator
            for y, row in enumerate(self.grid):
                symbol_row = " ".join(color_to_symbol.get(cell, ".") for cell in row)
                f.write(f"{y:3} | {symbol_row}\n")

        print(f"ASCII exported to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FrameEditor(root)
    root.mainloop()