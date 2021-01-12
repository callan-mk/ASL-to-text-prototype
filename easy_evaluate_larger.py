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
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

# Model file must contain already trained model!
MODEL_FILE = 'ASLID_large3.model'
model = load_model(MODEL_FILE)
this_dir = os.getcwd()
image_directory = 'ASL_evalsubset'
#TEST_IMG_FILE = os.path.join(this_dir, 'ASL_datasubset/C/C107.jpg')
HEIGHT = 299
WIDTH = 299
DEPTH = 'rgb'

eval_batch_size = 100
eval_steps = 50

# make class indices dict
source_dir = os.path.join(this_dir, image_directory)
class_names = os.listdir(source_dir)
class_names = sorted(class_names)
name_id_map = dict(zip(range(len(class_names)), class_names))


##--------------------------------------------------------------------------------
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
    preds = predict(model, this_img)
    classes = preds.argmax(axis = -1)
    class_names = name_id_map[classes]
    return classes, class_names

#import & organize image data...
def process_image_data(image_dir):
    image_datagen = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            )
    
    eval_generator = image_datagen.flow_from_directory(
            image_dir,
            target_size=(HEIGHT, WIDTH),
            color_mode = DEPTH,
            batch_size= eval_batch_size,
            )
    #process the images and return processed sets?
    return eval_generator



##--------------------------------------------------------------------------------
# Test accuracy of model on evalsubset.   
def eval_model(logfile="Model_Eval.txt"):
    f = open(logfile, "a")
    
    eval_gen = process_image_data(image_directory)
    
    eval_list = model.evaluate_generator(eval_gen,
                                         steps = eval_steps,
                                         max_queue_size=10,
                                         verbose=1)
    print('Display Labels for Scalar Outputs:', model.metrics_names)
    print('Display Labels for Scalar Outputs:', model.metrics_names, file = f)
    print(eval_list)
    print(eval_list, file = f)
    
    preds = model.predict_generator(eval_gen,
                                     steps = eval_steps,
                                     max_queue_size=10,
                                     verbose=1)
    print(preds)
    print(preds, file = f)
    f.close()
    return eval_list, preds
    
    