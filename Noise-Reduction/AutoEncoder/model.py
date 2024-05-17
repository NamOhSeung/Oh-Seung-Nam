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
from keras import layers
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
    hidden_1 = layers.Dense(hidden_dim, activation='relu')(input_img)
    code = layers.Dense(code_dim, activation='relu')(hidden_1)
    hidden_2 = layers.Dense(hidden_dim, activation='relu')(code)
    output_img = layers.Dense(input_dim, activation='sigmoid')(hidden_2)

    return keras.Model(input_img,output_img)
