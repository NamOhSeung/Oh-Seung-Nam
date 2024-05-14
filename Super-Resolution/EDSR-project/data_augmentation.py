import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers


AUTOTUNE = tf.data.experimental.AUTOTUNE


def flip_randomly(lowres_img, highres_img):
    """이미지를 무작위로 좌우로 뒤집는다."""

    # 0부터 1까지의 균일 분포에서 무작위 숫자를 생성
    rn = tf.random.uniform(shape=(), maxval=1)
    # 만약 rn이 0.5보다 작으면 원본 이미지를 반환하고, 그렇지 않으면 뒤집힌 이미지를 반환
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )


def rotate_randomly(lowres_img, highres_img):
    """Randomly rotates images by 90, 180, or 270 degrees."""

    # Generate a random integer from a uniform distribution in the range [0, 3]
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Rotate the images based on the random number
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    """Crop images.

    Low resolution images: 24x24
    High resolution images: 96x96
    """
    lowres_crop_size = hr_crop_size // scale  # 96//4=24
    lowres_img_shape = tf.shape(lowres_img)[:2]  # (height, width)

    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    )

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height : lowres_height + lowres_crop_size,
        lowres_width : lowres_width + lowres_crop_size,
    ]  # 24x24
    highres_img_cropped = highres_img[
        highres_height : highres_height + hr_crop_size,
        highres_width : highres_width + hr_crop_size,
    ]  # 96x96

    return lowres_img_cropped, highres_img_cropped


def prepare_dataset(dataset_cache, training=True):
    """Prepare a `tf.data.Dataset` object for training or validation."""

    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, scale=4),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(rotate_randomly, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_randomly, num_parallel_calls=AUTOTUNE)

    # Batch the data
    ds = ds.batch(16)

    if training:
        # Repeat the data to make the dataset infinite
        ds = ds.repeat()

    # Prefetch the data for better performance
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds



