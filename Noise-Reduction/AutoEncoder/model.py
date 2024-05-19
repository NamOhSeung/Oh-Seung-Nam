import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm
import os
import random

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
from glob import glob
from tensorflow._api.v2.nn import depth_to_space

import os
import matplotlib.pyplot as plt


def Autoencoder(x_train, hidden_dim, code_dim):

    input_dim = x_train.shape[-1]
    input_img = Input(shape=(input_dim, ))
    hidden_1 = Dense(hidden_dim, activation='relu')(input_img)
    code = Dense(code_dim, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_dim, activation='relu')(code)
    output_img = Dense(input_dim, activation='sigmoid')(hidden_2)

    return keras.Model(input_img,output_img)

# 모델 출력층의 activation function을 sigmoid로, 
# 컴파일 시 loss function을 binary crossentropy로 설정한 것에 주의하자. 
# 0과 1 사이로 구성되어 있는 픽셀 값의 특성을 살리기 위해 두 조합을 사용했다.


def autoencoder(batch_shape):
    # 모델 레이어 설정
    x_Input = Input(batch_shape=batch_shape) # 인코더 입력
    # 인코더
    e_conv = Conv2D(filters=10, kernel_size=(3, 3), strides=1, padding='SAME', activation='relu')(x_Input)
    e_pool = MaxPooling2D(pool_size=(2, 2), strides=1, padding='SAME')(e_conv)
    # 디코더
    d_conv = Conv2DTranspose(filters=10, kernel_size=(3, 3), strides=1, padding='SAME', activation='relu')(e_pool)
    x_Output = Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='SAME', activation='sigmoid')(d_conv)

    return Model(x_Input, x_Output)