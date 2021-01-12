# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:42:44 2019

Callan M Keller

ASL Alphabet to Text - WIP
"""

import os
import numpy as np
import cv2
import tensorflow.keras as keras

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

# Model file must contain already trained model!
MODEL_FILE = 'ASLID_large3.model'
model = load_model(MODEL_FILE)
this_dir = os.getcwd()

#image properties
HEIGHT = 299
WIDTH = 299

# make class indices dict
source_dir = os.path.join(this_dir, 'ASL_datasubset')
class_names = os.listdir(source_dir)
class_names = sorted(class_names)
name_id_map = dict(zip(range(len(class_names)), class_names))


#what camera input to use, 0 = local webcam
cam = 0

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

def load_image(img, h, w):
    #test_image = cv2.imread(img)
    test_image = cv2.resize(img, (h, w))
    test_image = test_image[...,::-1].astype(np.float32)
    return test_image

def predict_this(img_in):
    this_img = load_image(img_in, HEIGHT, WIDTH)
    preds = predict(model, this_img)
    classes = preds.argmax(axis = -1)
    class_names = name_id_map[classes]
    return class_names

    
def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('My Webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    
    
##-------------------------------------------------------------------------------------
def run_interpreter(logfile="Interpreted.txt"):
    f = open(logfile, "a")
    #show_webcam(mirror=True)
    
    cap = cv2.VideoCapture(cam)    
    while True:
        # Capture frame-by-frame
        ret_val, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #For capture image in monochrome
        rgbImage = frame #For capture the image in RGB color space

        # Display the resulting frame
        cv2.imshow('Webcam', rgbImage)
        
        #print('ready for image')
        
        #Wait to press 'c' key for capturing
        if cv2.waitKey(1) == ord('z'):
            #Show image
            cv2.imshow('Capture',rgbImage)
            #Format & Classify Image
            predicted = predict_this(rgbImage)
            if predicted == 'del':
                #delete last character? not sure how?
                pass
            elif predicted == 'nothing':
                pass
            elif predicted == 'space':
                pred = ' '
                print(pred)
                print(pred, file = f)
            else:
                pred = predicted
                print(pred)
                print(pred, file = f)
                pass
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        
    # When everything done, release the capture
    cap.release()
    f.close()
    cv2.destroyAllWindows()