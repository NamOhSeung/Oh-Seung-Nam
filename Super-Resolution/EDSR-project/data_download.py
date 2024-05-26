import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from data_augmentation import *
from model import *

from tensorflow import keras
from keras import layers


def div2k_download():

    div2k_data = tfds.image.Div2k(config="bicubic_x4",data_dir = "./image_dataset")
    div2k_data.download_and_prepare()

    # Taking train data from div2k_data object
    train = div2k_data.as_dataset(split="train", as_supervised=True) # as_supervised: 데이터셋을 (입력, 레이블) 쌍으로 반환할지 여부를 결정. True이면 쌍으로, False이면 입력 데이터만 반환
    train_cache = train.cache()
    # Validation data
    val = div2k_data.as_dataset(split="validation", as_supervised=True)
    val_cache = val.cache()

    return train_cache, val_cache
