import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from gallery import ImageGallery

prompt = "blue shirt boy walking in the port"
image_gallery_path = r"img"
model_name = "openai/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class SmartGalleryAgent():
    """SmartGalleryAgent class to interact"""
    def __init__(self, model_name=model_name, image_gallery_path=image_gallery_path, prompt=prompt):
        self.image_gallery = self.load_gallery(image_gallery_path)
        self.model, self.tokenizer, self.processor = self.load_model(model_name)
        self.image_features = self.get_image_features()
        self.text_features = self.get_text_features(prompt)

    def load_gallery(self, image_gallery_path):
        """Load the image gallery"""
        return ImageGallery(image_gallery_path)

    def load_model(self, model_name):
        """Load the CLIP model"""
        model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, tokenizer, processor

    def get_image_features(self):
        """Get the image features"""
        image_features = []
        for img_path in tqdm(self.image_gallery.imgs):
            img = Image.open(img_path).convert("RGB")
            image = self.processor(images=img, return_tensors="pt")['pixel_values'].to(device)
            image_emb = self.model.get_image_features(pixel_values=image)
            image_emb = image_emb.squeeze(0).cpu().detach().numpy()
            image_emb = image_emb / np.linalg.norm(image_emb)
            image_features.append(image_emb)
        return image_features

    def get_text_features(self, prompt):
        """Get the text features"""
        input = self.tokenizer(prompt, return_tensors="pt")
        text_emb = self.model.get_text_features(**input)
        text_emb = text_emb.cpu().detach().numpy()
        return text_emb


if __name__ == "__main__":

    img_folder_path = r"test"
    agent = SmartGalleryAgent(model_name, img_folder_path)
    # Tokenize the prompt
    text_emb = agent.text_features
    # Get the image features
    image = agent.image_features
    scores = [np.dot(text_emb, img_feat) for img_feat in agent.image_features]
    idx = np.argmax(scores)
    print(agent.image_gallery.imgs[idx])
    img = Image.open(agent.image_gallery.imgs[idx])
    img.show()
