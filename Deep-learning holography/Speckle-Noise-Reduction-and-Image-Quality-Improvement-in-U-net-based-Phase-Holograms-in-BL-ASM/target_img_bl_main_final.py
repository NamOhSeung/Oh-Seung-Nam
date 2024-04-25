import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm
import datetime
from propagation_functions import *
import torch
from data_function import data_preprocessing, data_load

# get_default_graph 같은 문제는 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD
from phase_optimization import *
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, mean_squared_error
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# parameter
wavelength_r = 633e-9
k_r = 2*np.pi/wavelength_r
pixel_size = 8.5e-6
shape = (512,512,1)

depth1 = 0.5
depth2 = 0.6
depth3 = 1
depth4 = 1.5
depth5 = 2

# train_path = './image_data/indoor_train_2d_phase_f06/'
train_path = './image_data/indoor_train/All_rgb(0~599)/'
# train_path = './image_data/DIV2K_train_HR_512/'
# train_path = './image_data/DIV2K_train_HR_756/'

# test_path = './image_data/Alpha/Alpha_amp_test_r/'
test_path = './image_data/indoor_train/All_rgb(0~599)/'
# test_path = './image_data/DIV2K_train_HR_512/'
# test_path = './image_data/DIV2K_train_HR_756/'


# x_train_path = './image_data/Alpha/Alpha_map_2D_phase_x_train'
# x_validate_path = './image_data/Alpha/Alpha_map_2D_phase_x_validate'
# y_train_path = './image_data/Alpha/Alpha_amp_y_train'
# y_validate_path= './image_data/Alpha/Alpha_amp_y_validate'


def U_Net(shape): # downsampling -> upsampling

    inputs = tf.keras.Input(shape=shape, name='target')
    conv1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same')(inputs) # 64
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(32, kernel_size=(3,3),activation='relu',padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    maxpooling1 = MaxPooling2D(pool_size=(2,2))(conv2) # 378

    conv3 = Conv2D(64, kernel_size=(3,3),activation='relu',padding='same')(maxpooling1) # 256
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(64, kernel_size=(3,3),activation='relu',padding='same')(conv3)
    conv4 = BatchNormalization()(conv3)
    maxpooling2 = MaxPooling2D(pool_size=(2,2))(conv4) # 189

    conv5 = Conv2D(128, kernel_size=(3,3),activation='relu',padding='same')(maxpooling2) # 256
    conv6 = Conv2D(128, kernel_size=(3,3),activation='relu',padding='same')(conv5)

    convtrans1 = Conv2DTranspose(64, kernel_size=(3,3),strides=(2,2),activation='relu',padding='same')(conv6) # 1024
    residual1 = concatenate([convtrans1,conv4]) # 378

    conv9 = Conv2D(32, kernel_size=(3,3),activation='relu',padding='same')(residual1) # 256
    conv10 = Conv2D(32, kernel_size=(3,3),activation='relu',padding='same')(conv9)

    convtrans2 = Conv2DTranspose(32, kernel_size=(3,3),strides=(2,2),activation='relu',padding='same')(conv10) #128
    residual2 = concatenate([convtrans2,conv2]) # 756

    conv11 = Conv2D(16, kernel_size=(3,3),activation='relu',padding='same')(residual2)
    conv12 = Conv2D(16, kernel_size=(3,3),activation='relu',padding='same')(conv11)
    
    phase_img = Conv2D(1, kernel_size=(3,3),activation='relu',padding='same')(conv12)
    
    model = Model(inputs=inputs, outputs=phase_img)
    
    return model


def recon_loss(y_true, y_pred):
    y_true = y_true[:,:,:,0] # angular spectrum 때만
    # y_pred = y_pred-tf.reduce_mean(y_pred)
    # y_pred = band_limited_angular_spectrum1(y_pred, k_r, -depth2, pixel_size, wavelength_r)
    # phase = tf.math.angle(y_pred)
    y_pred = band_limited_angular_spectrum4(y_pred, k_r, depth2, pixel_size, wavelength_r, 0.0001)
    err = y_true - y_pred
    loss = tf.math.reduce_mean(tf.math.abs(err))  # mae
    return loss


def main(): # main 함수 통해 코드 실행

  x_train, x_validate = data_load(train_path, 0.2) 
  y_train, y_validate = data_load(test_path, 0.2) 
  
  x_train = np.expand_dims(x_train[:,:,:,0],axis=-1) # (N,512,512,1)
  x_validate = np.expand_dims(x_validate[:,:,:,0],axis=-1)
  y_train = np.expand_dims(y_train[:,:,:,0],axis=-1)
  y_validate = np.expand_dims(y_validate[:,:,:,0],axis=-1)

  x_train = np.abs(x_train)/np.max(np.abs(x_train))
  x_validate = np.abs(x_validate)/np.max(np.abs(x_validate)) # int로 진행 
  y_train = np.abs(y_train)/np.max(np.abs(y_train))
  y_validate = np.abs(y_validate)/np.max(np.abs(y_validate))

  # x_train = x_train - np.mean(x_train)  
  # x_validate = x_validate - np.mean(x_validate)

  random = np.random.random((756,1344,1))*np.pi
  x_train = x_train*np.exp(1j*x_train)
  x_validate = x_validate*np.exp(1j*x_validate)

  x_train = band_limited_angular_spectrum3(x_train,k_r,-depth2,pixel_size,wavelength_r,0.0001)
  x_validate = band_limited_angular_spectrum3(x_validate,k_r,-depth2,pixel_size,wavelength_r,0.0001)

  x_train = tf.math.angle(x_train)
  x_validate = tf.math.angle(x_validate)

  shape = (756,1344,1)
  model = U_Net(shape)

  # run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
  # runmeta = tf.compat.v1.RunMetadata()

  cos_decay = tf.keras.experimental.CosineDecay(initial_learning_rate=0.0001, decay_steps=50, alpha=0.001)
  Adam = tf.keras.optimizers.Adam(learning_rate=cos_decay)
  model.compile(optimizer=Adam, loss=recon_loss) # image, phase [x_train, y_train는 따로 적용할 필요 없음]
  
  # save weight
  filename = './checkpoints/checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(5, 5) # from epoch  
  log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  
  # my custom callback
  callbacks = [ 
     
      ModelCheckpoint(filepath=filename,
                            monitor='val_loss',
                            save_weights_only=True, 
                            verbose=1,  
                            save_best_only=True, 
                            mode='auto') 
                          ]
  

  history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), 
            epochs=100, batch_size=8, callbacks=callbacks)

  print(history.history['loss'])
  print(history.history['val_loss'])
  fig, loss_ax = plt.subplots()

  loss_ax.plot(history.history['loss'],'y',label='train loss')
  loss_ax.plot(history.history['val_loss'],'r',label='val loss')
  loss_ax.set_xlabel('epoch')
  loss_ax.set_ylabel('loss')
  loss_ax.legend(loc='upper left')
  plt.show()

  model.save('./image_Data/models/U2_11_2d_img_e100_b8_lr4d_f06_band_angular_5_final.h5')

  
if __name__ == "__main__":
  main()
