import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import BackendFilter, backend_registry
from PIL import Image


class ResultViewer:
    """Show the search results in one window, left and right arrows to browse, q to close"""
    def __init__(self, prompt, results):
        self.prompt = prompt
        self.results = results
        self.index = 0
        # free the arrow keys, matplotlib binds them to the zoom history by default
        plt.rcParams["keymap.back"] = [key for key in plt.rcParams["keymap.back"] if key != "left"]
        plt.rcParams["keymap.forward"] = [key for key in plt.rcParams["keymap.forward"] if key != "right"]
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(prompt)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_key(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.results)
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.results)
        else:
            return
        self.draw()

    def draw(self):
        img_path, score = self.results[self.index]
        self.ax.clear()
        self.ax.imshow(Image.open(img_path).convert("RGB"))
        self.ax.set_title(f"{os.path.basename(img_path)}  {self.index + 1}/{len(self.results)}  score {score:.3f}")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def show(self):
        self.draw()
        plt.show()


def can_show():
    """False when matplotlib has no GUI backend, e.g. no display or tkinter missing"""
    return matplotlib.get_backend().lower() in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
