import numpy as np
import cv2
from keras.models import load_model
import process_image

model = load_model("./model/best_model/digit_classifier.h5")
image_path = "./image/before/sudoku-puzzle-863979.jpg"

digit = process_image.pre_process_images()

def extract_numbers(digit):
    prediction = list()
    for d in range(len(digit)):
        s = digit[d].reshape(28*28)
        black_pixel = s.shape[0]*0.81
        dig = digit[d].reshape(1,28,28,1)
        if black_pixel > sum(s<100):
            pred = model.predict_classes(dig,verbose = 0)
            pred = pred[0]
            prediction.append(pred)
        else:
            prediction.append('.')
    return prediction