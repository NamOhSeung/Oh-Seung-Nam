import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm
import os
import random

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
from glob import glob
from tensorflow._api.v2.nn import depth_to_space

 
# preprocessing => run.py와 data_utils에서 참고

'''
1. 고해상도 데이터셋에서 48*48 사이즈의 저해상도 패치와 고해상도 패치를 추출(normalize, flip 중간에 수행)
2. 
'''



# model 10시 45분까지 

def edsr(scale, num_filters: int = 256, num_res_blocks: int = 32):
    """
    Creates an EDSR model.
    
    Parameters
    ----------
    num_filters: int
        Number of filters per convolution layer.
        Default=64

    num_res_blocks: int 
        Number of residual blocks in the model
        Default=16 

    Returns
    -------
        EDSR Model object.
    """
    DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
    normalize = lambda x: (x - DIV2K_RGB_MEAN) / 127.5
    denormalize = lambda x: x * 127.5 + DIV2K_RGB_MEAN
    pixel_shuffle = lambda x: depth_to_space(x, 2) 

    def residual_block(layer_input, filters, res_scale = 1.0):
        """Residual block described in paper"""
        res = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(layer_input) # relu 넣어줌
        res = Conv2D(filters, kernel_size=3, strides=1, padding='same')(res)    
        res = Lambda(lambda res: res * res_scale)(res) # multi scaling 적용?
        res = concatenate([res, layer_input])
        return res

    # def upsample_block(layer_input, i) :
    #     u = Conv2D(num_filters*4, kernel_size=3, strides=1, padding='same', name=f"conv_up_{i}")(layer_input)
    #     u = Lambda(pixel_shuffle, name=f"pix_shuf_{i}")(u) # 람다 레이어에서 pixel shuffle
    #     return u 
    
    def upsample(x, scale, num_filters):
        def upsample_1(x, factor, **kwargs):
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            return Lambda(pixel_shuffle(scale=factor))(x)

        if scale == 2:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
        elif scale == 3:
            x = upsample_1(x, 3, name='conv2d_1_scale_3')
        elif scale == 4:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
            x = upsample_1(x, 2, name='conv2d_2_scale_2')

        return x

    input = Input(shape=(None, None, 3), name="LR Batch")
    x = Lambda(normalize)(input)

    x = r = Conv2D(num_filters, 3, padding='same', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
    # 가중치 규제를 통해서 과적합을 방지하고 모델의 일반화 성능을 향상시킴. 
    for i in range(num_res_blocks):
        r = residual_block(r, num_filters, i,  name="residual_block")

    c2 = Conv2D(num_filters, 3, padding='same', name="conv_out")(r)
    c2 = concatenate(name="add_out")([x, c2])

    u1 = upsample(c2, scale, num_filters)
    c3 = Conv2D(3, 3, padding='same', name="conv_final")(u1)

    x_out = Lambda(denormalize, name="denormalize_output")(c3)
    return Model(input, x_out, name="EDSR")

