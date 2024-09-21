import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import agent
from agent import SmartGalleryAgent, load_config, normalize, rank_images, resolve_device, to_json


def test_normalize_returns_unit_vector():
    assert np.isclose(np.linalg.norm(normalize(np.array([3.0, 4.0]))), 1.0)


def test_rank_images_orders_by_similarity():
    image_features = np.stack([normalize(np.array(v)) for v in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0])])
    text_emb = normalize(np.array([0.0, 1.0]))

    ranked = rank_images(text_emb, image_features, ["x.jpg", "y.jpg", "xy.jpg"], top_k=2)

    assert [path for path, _ in ranked] == ["y.jpg", "xy.jpg"]
    assert np.isclose(ranked[0][1], 1.0)


class FakeModel:
    """Mimics transformers 5 CLIPModel, features come back in pooler_output"""
    def get_text_features(self, input_ids, **kwargs):
        assert input_ids.device.type == "cpu"
        return SimpleNamespace(pooler_output=torch.tensor([[0.0, 2.0]], device=input_ids.device))


class FakeTokenizer:
    def __call__(self, prompt, return_tensors):
        return SimpleNamespace(to=lambda device: {"input_ids": torch.zeros(1, 3, dtype=torch.long, device=device)})


def test_search_uses_pooler_output_and_moves_text_to_device():
    gallery_agent = SmartGalleryAgent.__new__(SmartGalleryAgent)
    gallery_agent.device = "cpu"
    gallery_agent.model, gallery_agent.tokenizer = FakeModel(), FakeTokenizer()
    gallery_agent.image_gallery = SimpleNamespace(imgs=["x.jpg", "y.jpg"])
    gallery_agent.image_features = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert gallery_agent.search("anything") == [("y.jpg", 1.0)]


def test_load_config_reads_settings_json():
    with open(agent.SETTINGS_PATH) as f:
        assert load_config() == json.load(f)


def test_load_config_applies_file_then_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gallery": "photos", "top_k": 5}))

    config = load_config(str(config_path), top_k=3, gallery=None)

    assert config["gallery"] == "photos"
    assert config["top_k"] == 3


def test_load_config_rejects_unknown_settings(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"topk": 5}))

    with pytest.raises(ValueError, match="topk"):
        load_config(str(config_path))


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_to_json_lists_results_in_order():
    output = json.loads(to_json("a dog", [("y.jpg", 0.9), ("x.jpg", 0.4)]))

    assert output == {"prompt": "a dog", "results": [{"path": "y.jpg", "score": 0.9}, {"path": "x.jpg", "score": 0.4}]}
