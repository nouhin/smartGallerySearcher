import argparse

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from gallery import ImageGallery

prompt = "blue shirt boy walking in the port"
image_gallery_path = "img"
model_name = "openai/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"


class SmartGalleryAgent():
    """SmartGalleryAgent class to interact"""
    def __init__(self, model_name=model_name, image_gallery_path=image_gallery_path):
        self.image_gallery = self.load_gallery(image_gallery_path)
        self.model, self.tokenizer, self.processor = self.load_model(model_name)
        self.image_features = self.get_image_features()

    def load_gallery(self, image_gallery_path):
        """Load the image gallery"""
        return ImageGallery(image_gallery_path)

    def load_model(self, model_name):
        """Load the CLIP model"""
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, tokenizer, processor

    @torch.no_grad()
    def get_image_features(self):
        """Get the normalized image features"""
        image_features = []
        for img_path in tqdm(self.image_gallery.imgs):
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            image_emb = self.model.get_image_features(pixel_values=pixel_values).pooler_output
            image_features.append(normalize(image_emb.squeeze(0).cpu().numpy()))
        return np.stack(image_features)

    @torch.no_grad()
    def get_text_features(self, prompt):
        """Get the normalized text features"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        text_emb = self.model.get_text_features(**inputs).pooler_output
        return normalize(text_emb.squeeze(0).cpu().numpy())

    def search(self, prompt, top_k=1):
        """Return the top_k (image path, score) pairs matching the prompt"""
        text_emb = self.get_text_features(prompt)
        return rank_images(text_emb, self.image_features, self.image_gallery.imgs, top_k)


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
    parser.add_argument("--gallery", default=image_gallery_path)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    print(f"Using {device} device")
    agent = SmartGalleryAgent(model_name, args.gallery)
    for img_path, score in agent.search(args.prompt, args.top_k):
        print(f"{score:.3f}  {img_path}")
