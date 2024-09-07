from types import SimpleNamespace

import numpy as np
import torch

import agent
from agent import SmartGalleryAgent, normalize, rank_images


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
        assert input_ids.device.type == agent.device
        return SimpleNamespace(pooler_output=torch.tensor([[0.0, 2.0]], device=input_ids.device))


class FakeTokenizer:
    def __call__(self, prompt, return_tensors):
        return SimpleNamespace(to=lambda device: {"input_ids": torch.zeros(1, 3, dtype=torch.long, device=device)})


def test_search_uses_pooler_output_and_moves_text_to_device():
    gallery_agent = SmartGalleryAgent.__new__(SmartGalleryAgent)
    gallery_agent.model, gallery_agent.tokenizer = FakeModel(), FakeTokenizer()
    gallery_agent.image_gallery = SimpleNamespace(imgs=["x.jpg", "y.jpg"])
    gallery_agent.image_features = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert gallery_agent.search("anything") == [("y.jpg", 1.0)]
