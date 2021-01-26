import cv2
import numpy as np
import helper





def pre_process_images(image_path):
    
    image = helper.preprocess(image_path)
    crop = helper.crop_image(image)
    boxes = helper.grid_box(crop)
    digit = helper.get_digit(crop,boxes,28)
    return digit
