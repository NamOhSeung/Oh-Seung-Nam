import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.models import *
from keras import layers
from model import *

import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.optimizers import adam_v2
from keras.preprocessing import *
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers
from glob import glob
from tensorflow._api.v2.nn import depth_to_space
from function import *

# 하이퍼 파라미터
hidden_dim = 128
code_dim = 32
epochs = 50
learning_rate = 1e-4
batch_size = 300


# 데이터 불러오기
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# 랜덤 노이즈 추가
noise_factor = 0.3
x_train_noise = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noise = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noise = np.clip(x_train_noise, 0.0, 1.0) # 0~1 제한
x_test_noise = np.clip(x_test_noise, 0.0, 1.0)

# 정규화 및 차원 변경
x_train = x_train.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

x_train_noise = x_train_noise.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_test_noise = x_test_noise.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_train_noise = x_train_noise[:, :, :, np.newaxis]
x_test_noise = x_test_noise[:, :, :, np.newaxis]

# 파라미터 설정
n_height = x_train_noise.shape[1] # 28
n_width = x_train_noise.shape[2] # 28
n_channel = x_train_noise.shape[3] # 흑백: 1

batch_shape = (None, n_height, n_width, n_channel)

def main():

    #모델 구성
    model = autoencoder(batch_shape=batch_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.save('my_model')


    print('모델 구조')
    print(model.summary())
    hist = model.fit(x_train_noise, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    #화면에 이미지 그림
    def showImage(images):
        n = 0
        for k in range(2):
            plt.figure(figsize=(8, 2))
            for i in range(5):
                ax = plt.subplot(1, 5, i+1)
                plt.imshow(images[n])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                n += 1
            plt.show()

    # 노이즈 포함된 테스트 데이터 잡음 제거
    # 비교 위해 먼저 그림 그리기
    print("잡음 제거 전 테스트 데이터 10개")
    showImage(x_test_noise)
    # 학습된 모델 통과
    X_test_detected = model.predict(x_test_noise)
    # 잡음 제거 후 그림 그리기
    print("잡음 제거 후 테스트 데이터 10개")
    showImage(X_test_detected)


if __name__ == '__main__':
    main()
