import os

from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


class ImageGallery:
    def __init__(self, img_folder_path):
        self.img_folder_path = img_folder_path
        self.imgs = []
        self.load_imgs()

    def load_imgs(self):
        for img_name in sorted(os.listdir(self.img_folder_path)):
            if img_name.lower().endswith(IMG_EXTENSIONS):
                self.imgs.append(os.path.join(self.img_folder_path, img_name))

    def show_imgs(self):
        for img in self.imgs:
            img = Image.open(img)
            img.show()


if __name__ == "__main__":
    IMG_FOLDER_PATH = 'img'

    gallery = ImageGallery(IMG_FOLDER_PATH)
    print(gallery.imgs)
    print(gallery.img_folder_path)
