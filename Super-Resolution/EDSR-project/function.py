import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers



def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)
    return psnr_value

def psnr2(Original,compressed):
    mse = np.mean((Original-compressed)**2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    pixel_max = np.max(compressed).astype("uint8")
    PSNR = 20*np.log10(pixel_max / np.sqrt(mse))
    return PSNR


def SSIM(x, y):
    # assumption : x and y are grayscale images with the same dimension

    import numpy as np
    
    def mean(img):
        return np.mean(img)
        
    def sigma(img):
        return np.std(img)
    
    def cov(img1, img2):
        img1_ = np.array(img1[:,:], dtype=np.float64)
        img2_ = np.array(img2[:,:], dtype=np.float64)
                        
        return np.mean(img1_ * img2_) - mean(img1) * mean(img2)
    
    K1 = 0.01
    K2 = 0.03
    L = 256 # when each pixel spans 0 to 255
   
    C1 = K1 * K1 * L * L
    C2 = K2 * K2 * L * L
    C3 = C2 / 2
        
    l = (2 * mean(x) * mean(y) + C1) / (mean(x)**2 + mean(y)**2 + C1)
    c = (2 * sigma(x) * sigma(y) + C2) / (sigma(x)**2 + sigma(y)**2 + C2)
    s = (cov(x, y) + C3) / (sigma(x) * sigma(y) + C3)
        
    return l * c * s
