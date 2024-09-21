import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from gallery import ImageGallery

prompt = "blue shirt boy walking in the port"
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


class SmartGalleryAgent():
    """SmartGalleryAgent class to interact"""
    def __init__(self, model_name, image_gallery_path, device="cpu"):
        self.device = device
        self.image_gallery = self.load_gallery(image_gallery_path)
        self.model, self.tokenizer, self.processor = self.load_model(model_name)
        self.image_features = self.get_image_features()

    def load_gallery(self, image_gallery_path):
        """Load the image gallery"""
        return ImageGallery(image_gallery_path)

    def load_model(self, model_name):
        """Load the CLIP model"""
        model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, tokenizer, processor

    @torch.no_grad()
    def get_image_features(self):
        """Get the normalized image features"""
        image_features = []
        for img_path in tqdm(self.image_gallery.imgs):
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
            image_emb = self.model.get_image_features(pixel_values=pixel_values).pooler_output
            image_features.append(normalize(image_emb.squeeze(0).cpu().numpy()))
        return np.stack(image_features)

    @torch.no_grad()
    def get_text_features(self, prompt):
        """Get the normalized text features"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        text_emb = self.model.get_text_features(**inputs).pooler_output
        return normalize(text_emb.squeeze(0).cpu().numpy())

    def search(self, prompt, top_k=1):
        """Return the top_k (image path, score) pairs matching the prompt"""
        text_emb = self.get_text_features(prompt)
        return rank_images(text_emb, self.image_features, self.image_gallery.imgs, top_k)


def load_config(path=None, **overrides):
    """Load settings.json, then the optional config file on top, then the overrides that are not None"""
    with open(SETTINGS_PATH) as f:
        config = json.load(f)
    if path is not None:
        with open(path) as f:
            user_config = json.load(f)
        unknown = set(user_config) - set(config)
        if unknown:
            raise ValueError(f"Unknown settings in {path}: {', '.join(sorted(unknown))}")
        config.update(user_config)
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def to_json(prompt, results):
    return json.dumps({"prompt": prompt, "results": [{"path": path, "score": score} for path, score in results]}, indent=2)


def normalize(emb):
    return emb / np.linalg.norm(emb)


def rank_images(text_emb, image_features, img_paths, top_k=1):
    """Rank images by cosine similarity, embeddings are expected normalized"""
    scores = image_features @ text_emb
    best = np.argsort(scores)[::-1][:top_k]
    return [(img_paths[i], float(scores[i])) for i in best]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search an image gallery with a text prompt")
    parser.add_argument("prompt", nargs="?", default=prompt)
    parser.add_argument("--config", help="json file overriding settings.json")
    parser.add_argument("--gallery")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--json", action="store_true", help="print the results as json")
    args = parser.parse_args()

    config = load_config(args.config, gallery=args.gallery, top_k=args.top_k)
    device = resolve_device(config["device"])

    print(f"Using {device} device", file=sys.stderr)
    agent = SmartGalleryAgent(config["model_name"], config["gallery"], device)
    results = agent.search(args.prompt, config["top_k"])
    if args.json:
        print(to_json(args.prompt, results))
    else:
        for img_path, score in results:
            print(f"{score:.3f}  {img_path}")
