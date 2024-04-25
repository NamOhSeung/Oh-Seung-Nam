import numpy as np
import matplotlib.pyplot as plt
from Openholo.Openholo_python.ophpy.Depthmap import *
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm
import os
import random


import tensorflow as tf
from keras.preprocessing.image import img_to_array


def data_preprocessing(w,h,x_train_path,x_validate_path,y_train_path,y_validate_path):
    x_train = []
    files = os.listdir(x_train_path)
    for i in tqdm(files):
        img = cv2.imread(x_train_path+'/' + i)
        img = cv2.resize(img, (w, h))
        x_train.append(img_to_array(img))

    x_validate  = []
    files = os.listdir(x_validate_path)
    for i in tqdm(files):
        img = cv2.imread(x_validate_path+'/' + i)
        img = cv2.resize(img, (w, h))
        x_validate.append(img_to_array(img))

    y_train = []
    files = os.listdir(y_train_path)
    for i in tqdm(files):
        img = cv2.imread(y_train_path+'/' + i)
        img = cv2.resize(img, (w, h))
        y_train.append(img_to_array(img))

    y_validate  = []                    
    files = os.listdir(y_validate_path)
    for i in tqdm(files):
        img = cv2.imread(y_validate_path+'/' + i)
        img = cv2.resize(img, (w, h))
        y_validate.append(img_to_array(img))

    x_train = np.reshape(x_train, (len(x_train), 1024, 1024, 3))
    x_train = np.expand_dims(x_train[:,:,:,0],axis=-1) #(N, 756, 1344, 3) => (N, 756, 1344, 1)
    x_train = np.abs(x_train)/np.max(np.abs(x_train)) # 0~1

    x_validate = np.reshape(x_validate, (len(x_validate), 1024, 1024, 3))
    x_validate = np.expand_dims(x_validate[:,:,:,0],axis=-1) 
    x_validate = np.abs(x_validate)/np.max(np.abs(x_validate))

    y_train = np.reshape(y_train, (len(y_train), 1024, 1024, 3))
    y_train = np.expand_dims(y_train[:,:,:,0],axis=-1)
    y_train = np.abs(y_train)/np.max(np.abs(y_train))

    y_validate = np.reshape(y_validate, (len(y_validate), 1024, 1024, 3))
    y_validate = np.expand_dims(y_validate[:,:,:,0],axis=-1)
    y_validate = np.abs(y_validate)/np.max(np.abs(y_validate)) # 데이터 안에 nan이 들어가 있으면 loss도 nan이 나옴.
    
    return x_train, x_validate, y_train, y_validate


def data_preprocessing_756(w,h,x_train_path,x_validate_path,y_train_path,y_validate_path):
    x_train = []
    files = os.listdir(x_train_path)
    for i in tqdm(files):
        img = cv2.imread(x_train_path+'/' + i)
        img = cv2.resize(img, (w, h))
        x_train.append(img_to_array(img))

    x_validate  = []
    files = os.listdir(x_validate_path)
    for i in tqdm(files):
        img = cv2.imread(x_validate_path+'/' + i)
        img = cv2.resize(img, (w, h))
        x_validate.append(img_to_array(img))

    y_train = []
    files = os.listdir(y_train_path)
    for i in tqdm(files):
        img = cv2.imread(y_train_path+'/' + i)
        img = cv2.resize(img, (w, h))
        y_train.append(img_to_array(img))

    y_validate  = []                    
    files = os.listdir(y_validate_path)
    for i in tqdm(files):
        img = cv2.imread(y_validate_path+'/' + i)
        img = cv2.resize(img, (w, h))
        y_validate.append(img_to_array(img))

    x_train = np.reshape(x_train, (len(x_train), 756, 1344, 3))
    x_train = np.expand_dims(x_train[:,:,:,0],axis=-1) #(N, 756, 1344, 3) => (N, 756, 1344, 1)
    x_train = np.abs(x_train)/np.max(np.abs(x_train)) # 0~1

    x_validate = np.reshape(x_validate, (len(x_validate), 756, 1344, 3))
    x_validate = np.expand_dims(x_validate[:,:,:,0],axis=-1) 
    x_validate = np.abs(x_validate)/np.max(np.abs(x_validate))

    y_train = np.reshape(y_train, (len(y_train), 756, 1344, 3))
    y_train = np.expand_dims(y_train[:,:,:,0],axis=-1)
    y_train = np.abs(y_train)/np.max(np.abs(y_train))

    y_validate = np.reshape(y_validate, (len(y_validate), 756, 1344, 3))
    y_validate = np.expand_dims(y_validate[:,:,:,0],axis=-1)
    y_validate = np.abs(y_validate)/np.max(np.abs(y_validate)) # 데이터 안에 nan이 들어가 있으면 loss도 nan이 나옴.
    
    return x_train, x_validate, y_train, y_validate

def data_load(dir_path_in, ratio = None, shuffle = None):
    
    data_train = []
    data_val = []
    
    dir_list = os.listdir(dir_path_in) 
    
    if shuffle is True :
        random.shuffle(dir_list)
    
    def data_spilit(dir_list,ratio):
        n = int(len(dir_list) * (1-ratio))
        dir_list_train,dir_list_val = dir_list[:n],dir_list[n:]
        return dir_list_train,dir_list_val

    dir_list_train,dir_list_val = data_spilit(dir_list,ratio=ratio)

    for i in range(len(dir_list_train)):
        img = np.asarray(Image.open(os.path.join(dir_path_in + dir_list[i])))
        data_train.append(img)

    data_train = np.array(data_train)

    for i in range(len(dir_list_val)):
        img = np.asarray(Image.open(os.path.join(dir_path_in + dir_list[i])))
        data_val.append(img)

    data_val = np.array(data_val)

    return data_train,data_val
