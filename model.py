#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 05 00:22:45 2019

@author: kunal
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import (Flatten, Dense, Lambda, Dropout, Cropping2D,
                          Conv2D,BatchNormalization)
from sklearn.model_selection import train_test_split


img = pd.read_csv('driving_log.csv')

train, valid = train_test_split(img, test_size=0.2)


def generate_data(img, batch_size):
    """
    
    """
    img.iloc[:,0] = img.iloc[:,0].apply(lambda x: 'IMG/'+ x.split('\\')[-1])
    img.iloc[:,1] = img.iloc[:,1].apply(lambda x: 'IMG/'+ x.split('\\')[-1])
    img.iloc[:,2] = img.iloc[:,2].apply(lambda x: 'IMG/'+ x.split('\\')[-1])
    
    path = []
    path.append(np.array(img.iloc[:,1]))
    path.append(np.array(img.iloc[:,0]))
    path.append(np.array(img.iloc[:,2]))
    path.append(np.array(img.iloc[:,3]))
    
    path = np.array(path).T
    
    total_batch =  len(path)
    new_batch_size = (batch_size//3)//2
    
    correction = 0.17
    
    for offset in range(0, total_batch, new_batch_size):
        batch_path = path[offset:offset+new_batch_size]
        
        images = []
        angles = []
        for i in batch_path:
            left_image = plt.imread(i[0])
            center_image = plt.imread(i[1])
            right_image = plt.imread(i[2])
            
            flipped_left_image = np.fliplr(left_image)
            flipped_right_image = np.fliplr(right_image)
            flipped_center_image = np.fliplr(center_image)
            
            steering_center = i[3]
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            flipped_center = -1*steering_center
            flipped_left = -1*steering_left
            flipped_right = -1*steering_right
            
            images.extend([left_image, center_image, right_image])
            images.extend([flipped_left_image, flipped_center_image, flipped_right_image])
            angles.extend([steering_left, steering_center, steering_right])
            angles.extend([flipped_left, flipped_center, flipped_right])
        
        yield np.array(images), np.array(angles)


######## creating model#############
model = Sequential()
model.add(Lambda(lambda x : (x/255)-0.5, input_shape = (160,320,3)))
#add cropping layer
model.add(Cropping2D(cropping = ((70,25),(0,0))))
#add convolution layer with filter size as 24, patch size of 5*5 and stride of 2*2 with relu activation
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())
#add convolution layer with filter size as 36, patch size of 5*5 and stride of 2*2 with relu activation
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(BatchNormalization())
#add convolution layer with filter size as 48, patch size of 5*5 and stride of 2*2 with relu activation
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(BatchNormalization())
#add convolution layer with filter size as 64, patch size of 3*3 and stride of 1*1 with relu activation
model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1)))
model.add(BatchNormalization())
#add convolution layer with filter size as 64, patch size of 3*3 and stride of 1*1 with relu activation
model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1)))
model.add(BatchNormalization())
#flatten the output from thee previous layer to pass to fully connected layer
model.add(Flatten())
#add fully connected layer with 1164 neurons
model.add(Dense(1164, activation='relu'))
#add fully connected layer with 100 neurons
model.add(Dense(100))
#add fully connected layer with 50 neurons
model.add(Dense(50))
#add fully conncected layer with 10 neurons
model.add(Dense(10))
model.add(Dropout(0.5))
#add output layer with one neuron as we are dealing with a regression problem
model.add(Dense(1))
######################################################


train_generator = generate_data(train, 30)
valid_generator = generate_data(valid, 30)


N_EPOCHS = 8
BATCH_SIZE = 30
N_IMAGES_PER_ROW = 6 # total 3 image per row * 2(inverse of all 3 images)
TRAINING_SAMPLES_PERCENT = 0.8 #80%
VALID_SAMPLES_PERCENT = 0.2

steps_per_epoch = int(((len(img)*TRAINING_SAMPLES_PERCENT)*N_IMAGES_PER_ROW)//(BATCH_SIZE*N_EPOCHS))
validation_steps = int(((len(img)*VALID_SAMPLES_PERCENT)*N_IMAGES_PER_ROW)//(BATCH_SIZE*N_EPOCHS))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch, validation_data=valid_generator, validation_steps=validation_steps, nb_epoch=N_EPOCHS)

model.save('model.h5')