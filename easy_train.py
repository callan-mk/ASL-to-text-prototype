# -*- coding: utf-8 -*-
"""
Callan M Keller

ASL Alphabet to Text - WIP

The program applies Transfer Learning to an existing model and re-trains it to classify a new set of images.
This takes an Inception v3 architecture model trained on ImageNet images,
and trains a new top layer that can recognize other classes of images.

The argument image_dir refers to a dataset, organized in a folder containing subfolders of images.
The label for each image is taken from the name of the subfolder it's in.
The model should be applicable to any dataset organized in this manner.

Required Packages: tensorflow, keras?, numpy, pillow, matplotlib, sklearn?, pandas?
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
#InceptionV3 is NOT A SEQUENTIAL MODEL
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dropout
from keras import applications  
from keras.utils.np_utils import to_categorical  

#image formatting?
from IPython.display import display, Image

#Import Inception v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input


    #Input Parameters:
#BOTTLENECK_TENSOR_SIZE = 2048

# dimensions of input images
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 'rgb'

#MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

MODEL_FILE = 'ASLID.model'

image_directory = 'ASL_datasubset'
training_steps = 250
training_epochs = 8
val_steps = 200
learning_rate = 0.01

dropout_rate = 0.3

testing_percentage = .10
validation_percentage = .10
train_batch_size = 100
#test_batch_size = -1
validation_batch_size = 50

# Image augmentation: no rotation, as orientation is part of ASL language syntax!
if_zca_whitening = False
if_flip_horizontal = True
width_shift = 0.1
height_shift = 0.1
rescale_val = 1. / 255.
#random_brightness = tuple?


##--------------------------------------------------------------------------------------
## Main function WIP?
def run_model():
    # Process Images
    train_gen, val_gen = process_image_data(image_directory, validation_percentage)
    classes = num_classes(image_directory)
    
    
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
      
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer - bottleneck of size 2048?
    x = Dense(2048, activation='relu')(x)
    # add a dropout layer, maybe improve accuracy?
    x = Dropout(dropout_rate)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(classes, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    """
    while True:
        try:
    """
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model on the new data for a few epochs
    model.fit_generator(train_gen,
                        steps_per_epoch = training_steps, 
                        epochs = training_epochs, 
                        verbose = 1, 
                        callbacks = None, 
                        validation_data = val_gen, 
                        validation_steps = val_steps, 
                        #validation_freq = 1, 
                        class_weight = None,
                        initial_epoch = 0)
    # Below: fine-tuning convolutional layers.
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    
    # we chose to train the top 2 inception blocks, 
    # i.e.: freeze the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_gen,
                        steps_per_epoch = training_steps, 
                        epochs = training_epochs, 
                        verbose = 1, 
                        callbacks = None, 
                        validation_data = val_gen, 
                        validation_steps = val_steps, 
                        #validation_freq = 1, 
                        class_weight = None,
                        initial_epoch = 0)
        #except (KeyboardInterrupt, SystemExit):
            #print("Wait")
            #save progress
            #input()  #wait for user to hit enter
    model.save(MODEL_FILE)


##--------------------------------------------------------------------------------------
## Component Functions

#USE ImageDataGenerator CLASS
    #should probably (hopefully) be able to use this to merge several functions...

#import & organize image data...
def process_image_data(image_dir, validation_percentage):
    image_datagen = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            validation_split = validation_percentage,
            #testing_split = testing_percentage,
            
            zca_whitening = if_zca_whitening,
            
            width_shift_range = width_shift,
            height_shift_range = height_shift,
            #brightness_range = random_brightness
            horizontal_flip = if_flip_horizontal,
            
            #rescale = rescale_val,
            )
    
    train_generator = image_datagen.flow_from_directory(
            image_dir,
            target_size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
            color_mode = MODEL_INPUT_DEPTH,
            batch_size= train_batch_size,
            #class_mode= idk?
            subset = "training",
            )
    
    validation_generator = image_datagen.flow_from_directory(
            image_dir,
            target_size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
            color_mode = MODEL_INPUT_DEPTH,
            batch_size= validation_batch_size,
            #class_mode= idk?,
            subset = "validation",
            )
    #process the images and return processed sets?
    return train_generator, validation_generator

#count number of classes in data set
def num_classes(image_dir):
    path = os.path.join(os.getcwd(), image_dir)
    classes = sum(os.path.isdir(os.path.join(path, i)) for i in os.listdir(path))
    return classes



