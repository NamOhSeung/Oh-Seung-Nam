# tensorflow 2.8
# python 3.10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm

import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, mean_squared_error
from keras.preprocessing import *
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers

import cv2
import os
from glob import glob
import random
import argparse
from preprocess import *
import time

d = 56
s = 12
m = 4

# d = 32
# s = 5
# m = 1

scale = 2
req_size = 20
inp_size = int(req_size/scale)
stride = 15

upscaling_factor = 4
color_channels = 3
batch_size = 32
epochs = 500
lr = 0.001
color_channels = 3
shape = (None,None,color_channels)


img_path1 = './image_data/yang91/'
img_path2 = './image_data/yang91/'

# def FSRCNN(shape):
#     initializer = initializers.HeNormal()
#     inputs = tf.keras.Input(shape=shape, name='input')
#     conv1 = Conv2D(filters=d, kernel_size=5, kernel_initializer=initializer, bias_initializer='zeros', padding= 'same')(inputs) # feature_extraction
#     PReLU1 = PReLU(alpha_initializer='zeros',shared_axes=[1,2])(conv1)
#     conv2 = Conv2D(filters=s, kernel_size=1, kernel_initializer=initializer, bias_initializer='zeros', padding= 'same')(PReLU1) # shrinking
#     PReLU2 = PReLU(alpha_initializer='zeros',shared_axes=[1,2])(conv2)
#     conv3 = Conv2D(filters=s, kernel_size=3, kernel_initializer=initializer, bias_initializer='zeros', padding= 'same')(PReLU2) # non_linear_mapping
#     conv4 = Conv2D(filters=s, kernel_size=3, kernel_initializer=initializer, bias_initializer='zeros', padding= 'same')(conv3) # non_linear_mapping
#     PReLU3 = PReLU(alpha_initializer='zeros',shared_axes=[1,2])(conv4)
#     conv5 = Conv2D(filters=d, kernel_size=1, bias_initializer='zeros', padding= 'same')(PReLU3) # expanding
#     PReLU4 = PReLU(alpha_initializer='zeros',shared_axes=[1,2])(conv5)
#     conv6 = Conv2DTranspose(filters=color_channels, kernel_size=9, strides=(2,2) ,kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.001), bias_initializer='zeros', padding= 'same')(PReLU4) # deconvolution

#     model = Model(inputs=inputs, outputs=conv6)

#     return model

def FSRCNN(shape):
    initializer = initializers.HeNormal()
    inputs = tf.keras.Input(shape=shape, name='input')
    conv1 = Conv2D(filters=d, kernel_size=5, kernel_initializer=initializer, bias_initializer='zeros', padding= 'same')(inputs) # feature_extraction
    PReLU1 = PReLU(alpha_initializer='zeros',shared_axes=[1,2])(conv1)
    conv2 = Conv2D(filters=s, kernel_size=1, kernel_initializer=initializer, bias_initializer='zeros', padding= 'same')(PReLU1) # shrinking
    PReLU2 = PReLU(alpha_initializer='zeros',shared_axes=[1,2])(conv2)
    conv3 = conv2
    for _ in range(m):
        conv3 = Conv2D(filters=s, kernel_size=3, kernel_initializer=initializer, bias_initializer='zeros', padding='same')(conv3) # non-linear mapping
        conv3 = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(conv3)
    conv4 = Conv2D(filters=d, kernel_size=1, bias_initializer='zeros', padding='same')(conv3) # expanding
    PReLU3 = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(conv4)
    conv5 = Conv2DTranspose(filters=color_channels, kernel_size=9, strides=(2, 2), kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.001), bias_initializer='zeros', padding='same')(PReLU3) # deconvolution
    model = tf.keras.Model(inputs=inputs, outputs=conv5)

    return model

def main():

    input, labels = preprocessing(img_path1, img_path2, req_size, inp_size, stride)

    input = np.asarray(input)
    labels = np.asarray(labels)

    input = input/255
    labels = labels/255

    model = FSRCNN(shape)

    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    checkpoint = ModelCheckpoint('original_fsrcnn_epoch500.h5',  
                             monitor='val_loss', 
                             save_weights_only=False,
                             verbose=0, 
                             save_best_only= True, 
                             mode='auto') 

    start_time = time.time()

    history = model.fit(input,labels,batch_size=batch_size,shuffle=True,validation_split=0.01 ,epochs=epochs,callbacks=[checkpoint], verbose=1)

    end_time = time.time()

    # 총 학습 시간 계산
    training_time = end_time - start_time
    print("총 학습 시간:", training_time, "초")

    
    print(history.history['loss'])
    print(history.history['val_loss'])
    fig, loss_ax = plt.subplots()

    loss_ax.plot(history.history['loss'],'y',label='train loss')
    loss_ax.plot(history.history['val_loss'],'r',label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()

