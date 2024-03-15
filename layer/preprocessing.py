import os

from PIL import Image
import numpy as np


class Preprocessing:
    def __init__(self):
        pass

    def resize(self, image):
        return image.reshape((3, 512, 512))

    def load_image(self, image_dir):
        img_list = os.listdir(image_dir)
        img_list_jpg = [img for img in img_list if img.endswith("jpg")]

        img_list_np = []

        for i in img_list_jpg:
            img = Image.open(image_dir + i)
            img_array = np.array(img)
            img_list_np.append(self.resize(img_array))

        return np.array(img_list_np)
