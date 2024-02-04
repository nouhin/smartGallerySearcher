import os

from PIL import Image


class ImageGallery:
    def __init__(self, img_folder_path):
        self.img_folder_path = img_folder_path
        self.imgs = []
        self.load_imgs()

    def load_imgs(self):
        for img_name in os.listdir(self.img_folder_path):
            img_path = os.path.join(self.img_folder_path, img_name)
            self.imgs.append(img_path)

    def show_imgs(self):
        for img in self.imgs:
            img = Image.open(img)
            img.show()


if __name__ == "__main__":
    IMG_FOLDER_PATH = 'test'

    gallery = ImageGallery(IMG_FOLDER_PATH)
    print(gallery.imgs)
    print(gallery.img_folder_path)
