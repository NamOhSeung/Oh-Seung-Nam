import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, mean_squared_error
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from function import *

import cv2
import os
from glob import glob
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

filter1, filter2, filter3 = 128, 64, 3
kernel_s1, kernel_s2, kernel_s3 = 9,3,5
upscale_factor = 3

input_size = 33
output_size = input_size - kernel_s1 - kernel_s2 - kernel_s3 + 3 # 19
shape = (33,33,3)
pad = abs(input_size - output_size) // 2 # 7
stride = 14

batch_size = 128
epochs = 200

# img path & weight path
path = "./image_data/T91"
save_path = "./model/SRCNN/SRCNN_200EPOCHS.h5"


def SRCNN(shape):
    initializer = initializers.GlorotNormal()

    inputs = tf.keras.Input(shape=shape, name='input')
    conv1 = Conv2D(filters=filter1, kernel_size=kernel_s1, activation='relu', kernel_initializer=initializer, bias_initializer='zeros')(inputs)
    conv2 = Conv2D(filters=filter2, kernel_size=kernel_s2, activation='relu', kernel_initializer=initializer, bias_initializer='zeros')(conv1)
    conv3 = Conv2D(filters=filter3, kernel_size=kernel_s3, activation='linear', kernel_initializer=initializer, bias_initializer='zeros')(conv2)

    model = Model(inputs=inputs, outputs=conv3)

    return model


def main():

    # image load
    img_paths = glob(path + "/" + "*.png")
    
    # image preprocessing
    sub_lr_imgs, sub_hr_imgs = image_preprocessing(img_paths=img_paths, upscale_factor=upscale_factor, input_size=input_size, output_size=output_size, stride=stride, pad=pad)

    # model configuration
    model = SRCNN(shape)

    filename = './model/SRCNN/SRCNN_128batch_200epoch.h5' # save path
    callbacks = [ 
        
        ModelCheckpoint(filepath=filename,
                            monitor='val_loss', 
                            save_weights_only=True, 
                            verbose=1, 
                            save_best_only=False,  
                            mode='auto')  
                            ]
    
    optimizer = Adam(lr = 0.0003)

    model.compile(optimizer = optimizer, loss ='MSE', metrics = ['MSE'])
    model.fit(sub_lr_imgs, sub_hr_imgs, batch_size = batch_size, epochs = epochs, verbose=1, callbacks = callbacks)

if __name__ == '__main__':
    main()



