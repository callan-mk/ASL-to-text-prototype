# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:17:30 2019

Callan M Keller

ASL Alphabet to Text - WIP

"""

import os
import numpy as np
import tensorflow.keras as keras

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

# Model file must contain already trained model!
MODEL_FILE = 'ASLID_large3.model'
this_dir = os.getcwd()
TEST_IMG_FILE = os.path.join(this_dir, 'ASL_datasubset/C/C107.jpg')
HEIGHT = 299
WIDTH = 299

# make class indices dict
source_dir = os.path.join(this_dir, 'ASL_datasubset')
class_names = os.listdir(source_dir)
class_names = sorted(class_names)
name_id_map = dict(zip(range(len(class_names)), class_names))


##--------------------------------------------------------------------------------
#creates and returns full path for the path of a given image
def to_dir(new_dir):
    y = os.path.join(this_dir, 'ASL_datasubset/C/C107.jpg')
    return y

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

#img = image.load_img(TEST_IMG_FILE, target_size=(HEIGHT, WIDTH))
def load_image(img_dir, h, w):
    this_img = image.load_img(img_dir, target_size=(h, w))
    return this_img
    

#assumes img is always 299x299
#test: predict_this(TEST_IMG_FILE)
def predict_this(img_dir):
    this_img = load_image(img_dir, HEIGHT, WIDTH)
    preds = predict(load_model(MODEL_FILE), this_img)
    classes = preds.argmax(axis = -1)
    class_names = name_id_map[classes]
    return classes, class_names
