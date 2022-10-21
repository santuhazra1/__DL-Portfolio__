import numpy as np
import onnxruntime
from PIL import Image


class Model:

    def __init__(self, model_path):
        self.model_path = model_path
        self._model = onnxruntime.InferenceSession(self.model_path)
    
    def resize(self, img, size):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), 2)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), 2)
    
    def center_crop(self, img, size):
        image_width, image_height = img.size
        crop_height, crop_width = size, size

        crop_top = int((image_height - crop_height + 1) * 0.5)
        crop_left = int((image_width - crop_width + 1) * 0.5)

        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
    
    def normalize(self, img):
        img = np.array(img, dtype=np.float32) / 255
        img -= 0.5
        img /= 0.5
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    
    def load_image(self, image):
        image = self.resize(image, 128)
        image = self.center_crop(image, 128)
        image = self.normalize(image)
        return image
    
    def __call__(self, image):
        image = self.load_image(image)
        model_input = {self._model.get_inputs()[0].name: image}
        output = self._model.run(None, model_input)[0][0]
        output = np.transpose(output, (1, 2, 0))
        return Image.fromarray((output * 255).astype(np.uint8))
