from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from viewer import ResultViewer, can_show


def make_viewer(tmp_path):
    results = []
    for i, name in enumerate(["a.jpg", "b.png", "c.jpg"]):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
        results.append((str(tmp_path / name), 0.3 - i * 0.1))
    viewer = ResultViewer("a dog", results)
    viewer.draw()
    return viewer


def test_arrows_browse_and_wrap_around(tmp_path):
    viewer = make_viewer(tmp_path)

    viewer.on_key(SimpleNamespace(key="left"))
    assert viewer.ax.get_title() == "c.jpg  3/3  score 0.100"

    viewer.on_key(SimpleNamespace(key="right"))
    viewer.on_key(SimpleNamespace(key="right"))
    assert viewer.ax.get_title() == "b.png  2/3  score 0.200"
    plt.close(viewer.fig)


def test_other_keys_are_ignored(tmp_path):
    viewer = make_viewer(tmp_path)

    viewer.on_key(SimpleNamespace(key="x"))

    assert viewer.index == 0
    plt.close(viewer.fig)


def test_arrow_keys_are_not_bound_to_zoom_history(tmp_path):
    viewer = make_viewer(tmp_path)

    assert "left" not in plt.rcParams["keymap.back"]
    assert "right" not in plt.rcParams["keymap.forward"]
    plt.close(viewer.fig)


def test_can_show_is_false_without_gui_backend():
    assert not can_show()
