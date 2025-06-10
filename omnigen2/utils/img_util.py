import numpy as np
from PIL import Image


def resize_image(image, max_pixels, img_scale_num):
    width, height = image.size
    cur_pixels = height * width
    ratio = (max_pixels / cur_pixels) ** 0.5
    ratio = min(ratio, 1.0) # do not upscale input image

    new_height, new_width = int(height * ratio) // img_scale_num * img_scale_num, int(width * ratio) // img_scale_num * img_scale_num

    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    return image
