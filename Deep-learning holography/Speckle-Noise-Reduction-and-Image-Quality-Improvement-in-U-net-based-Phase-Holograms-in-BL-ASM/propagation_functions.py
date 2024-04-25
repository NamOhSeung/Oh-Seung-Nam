# 2D_phase_main5

import numpy as np
import matplotlib.pyplot as plt
from Openholo.Openholo_python.ophpy.Depthmap import *
import matplotlib.image as mpimg
from PIL import Image
import cv2
from tqdm import tqdm
import datetime
import tensorflow as tf
import torch
import math

def complex_phase(phi_slm): # 맞춤형 위상형 패턴인 Pslm
    i_phi_slm = tf.dtypes.complex(np.float32(0.), phi_slm)
    return tf.math.exp(i_phi_slm)

# deepcgh 응용
def __prop__(cf_slm, H = None, center = False): # prop에서 H를 이용함. ASM 대신 이걸 쓴다면?
    if not center:
        H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm)) # (756,1344)->(1,756,1344)
        cf_slm *= tf.signal.fftshift(H, axes = [1, 2]) # H를 fftshift 시키고 cf_slm에 곱합으로서 lens 적용
    fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes = [1, 2])), axes = [1, 2])
    img = tf.cast(tf.expand_dims(tf.abs(tf.pow(fft, 2)), axis=-1), dtype=tf.dtypes.float32)
    return img


def lens_fn(shape,z,k,pixel_size,wavelength): # 매번 더해지는 이 phase가 업뎃의 중요한 역할을 담당함.
    
    ''' 1)Add a thin lens with a focal length equal to z
        
        2)angular spectrum 
    '''

    nv, nu = shape[0],shape[1] # 1024,1024
    x = np.linspace(-nu/2*pixel_size, nu/2*pixel_size, nu)
    y = np.linspace(-nv/2*pixel_size, nv/2*pixel_size, nv)
    X, Y = np.meshgrid(x, y)
    Z = X**2+Y**2
    lens_fn = tf.exp(1j*k*z)/(1j*wavelength*z)*tf.exp(1j*k/2/z*Z)
    lens_fn = tf.signal.fft2d(tf.signal.fftshift(lens_fn))*pixel_size**2
    lens_fn = tf.cast(lens_fn,dtype=tf.complex64) # (1024,1024)
    
    return lens_fn #[N,H,W]

def lens_fn2(shape,z,k,pixel_size,wavelength): # 매번 더해지는 이 phase가 업뎃의 중요한 역할을 담당함.
    nv, nu = shape[0],shape[1] #756,1344
    x = np.linspace(-nu/2*pixel_size, nu/2*pixel_size, nu)
    y = np.linspace(-nv/2*pixel_size, nv/2*pixel_size, nv)
    X, Y = np.meshgrid(x, y)
    Z = X**2+Y**2
    lens_fn = tf.exp(1j*k*z)/(1j*wavelength*z)*tf.exp(1j*k/2/z*Z)
    # lens_fn = 1./(1j*wavelength*z)*tf.exp(1j*k*(z+Z/2/z))
    lens_fn = tf.signal.fft2d(tf.signal.fftshift(lens_fn,axes=[0,1]))*pixel_size**2
    lens_fn = tf.cast(lens_fn,dtype=tf.complex64)
    lens_fn = tf.expand_dims(lens_fn,axis=0)
    lens_fn = tf.expand_dims(lens_fn,axis=-1) #(N,756,1344,C)

    return lens_fn

def phase_generate(x,lens_fn): # (N,1024,1024,1) , (N,1024,1024,1)
    complex_field = complex_field_fn(x) # (N,756,1344)
    complex_holo = ASM1(complex_field,lens_fn) # (N,756,1344)
    slm_phase = __phase__(complex_holo) # complex-> float (N,756,1344,1)

    return slm_phase


def complex_field_fn(amp): 

    amp = tf.cast(amp,tf.complex64)
    constant_phase = tf.exp(1j*amp)*np.pi
    complex_field = amp*constant_phase

    return complex_field


def ASM1(field,lens_fn): # complex_field, lens function
    
    field = tf.cast(field,dtype=tf.complex64) # (N,756,1344,1)  # 이걸 128로 하면 recon loss의 input y가 complex128이 됨.
    # field = tf.expand_dims(field,axis=-1)
    U1 = tf.signal.fft2d(tf.signal.fftshift(field,axes=[1,2])) # deepcgh
    U2 = lens_fn*U1 # unsupported
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2),axes=[1,2])
    

    return result        


def __phase__(field):
    phase = tf.math.angle(field) # float
    phase = tf.expand_dims(phase,axis=-1) #(N,756, 1344,1)
    phase = tf.cast(phase, dtype=tf.float64) # 오류 안 사라짐. 3번 spherical 할때는 사용하고 나머지는 안해도 됨.
    
    return phase


def ASM2(field,lens_fn): # phase, lens function
    field = tf.cast(field,dtype=tf.complex64)
    U1 = tf.signal.fft2d(tf.signal.fftshift(field,axes=[1,2]))
    U2 = lens_fn*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2),axes=[1,2])
    result = tf.math.abs(result)
    return result


def ASM5(field,lens_fn): # phase, lens function
    field = tf.cast(field,dtype=tf.complex64) # (N,756, 1344) # 이걸 128로 하면 recon loss의 input y가 complex128이 됨.
    U1 = tf.signal.fft2d(tf.signal.fftshift(field,axes=[1,2])) # deepcgh
    U2 = lens_fn*U1
    recon = tf.signal.ifftshift(tf.signal.ifft2d(U2),axes=[1,2])
    recon = tf.abs(recon)/tf.math.reduce_max(tf.abs(recon))

    return recon

# 일단 u-net에서 건드리는건 맞고 가중치로 움직일 변수를 어떤걸로 해야할지 생각해야함. spherical phase도 wavelength에 따라 loss값이 천자만별이 됨.

def angular_spectrum1(field, k, distance, dx, wavelength):
    nv, nu = field.shape[1],field.shape[2] # 756, 1344
    x = np.linspace(-nu/2*dx, nu/2*dx, nu)
    y = np.linspace(-nv/2*dx, nv/2*dx, nv)
    X, Y = np.meshgrid(x, y)
    Z = X**2+Y**2
    h = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/(2*distance)*Z)
    h = tf.signal.fft2d(tf.signal.fftshift(h))*dx**2
    h = tf.expand_dims(h, axis=0)
    h = tf.expand_dims(h, axis=-1) # (756, 1344,1) # 이것도 모델 마지막 shape에 영향 감. 첫번째에 영향가는듯? 없애니깐 첫번째인 [1,756,1344]이 [756,1344] 된 듯?
    h = tf.cast(h, dtype=tf.complex64) 
    field = tf.cast(field,dtype=tf.complex64) # (N,756,1344,1) # 이걸 64로 하면 recon loss의 input y가 complex64이 됨.
    U1 = tf.signal.fft2d(tf.signal.fftshift(field,axes=[1,2])) # deepcgh
    U2 = h*U1
    complex_holo = tf.signal.ifftshift(tf.signal.ifft2d(U2),axes=[1,2])

    return complex_holo # complex type


def angular_spectrum2(field, k, distance, dx, wavelength): # [N,1024,1024]
    field  = field - tf.reduce_mean(field) # (N,1024,1024,1)
    nv, nu = field.shape[1],field.shape[2] # 1024, 1024
    x = np.linspace(-nu/2*dx, nu/2*dx, nu)
    y = np.linspace(-nv/2*dx, nv/2*dx, nv)
    X, Y = np.meshgrid(x, y)
    Z = X**2+Y**2
    h = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/(2*distance)*Z) #[1024,1024]
    h = tf.signal.fft2d(tf.signal.fftshift(h))*dx**2
    h = tf.expand_dims(h, axis=0) # (1,1024, 1024)
    # h = tf.expand_dims(h, axis=-1) # (1,1024, 1024, 1)
    h = tf.cast(h, dtype=tf.complex64) 
    field = tf.cast(field,dtype=tf.complex64) # (N,1024,1024) 
    field = field[:,:,:,0] #[N,1024,1024,1] =>[N,1024,1024]
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # deepcgh
    U2 = h*U1 # [1,1024,1024], [N,1024,1024].
    recon = tf.signal.ifftshift(tf.signal.ifft2d(U2))
    recon = tf.math.abs(recon)

    return recon


def band_limited_angular_spectrum1(field, k, distance, dx, wavelength, sampling_factor):

    nv, nu = field.shape[1],field.shape[2] # (756,1344)
    x = np.linspace(-nu*sampling_factor/2*dx, nu-sampling_factor/2*dx, nu) # x = (1344,)
    y = np.linspace(-nv+sampling_factor/2*dx, nv-sampling_factor/2*dx, nv) # y = (756,)
    X, Y = np.meshgrid(x, y) # X, Y 모두 (756, 1344)
    Z = X**2+Y**2 # 원형 모양 
    h = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h = np.fft.fft2(np.fft.fftshift(h))*dx**2 # 이미지 영역이 주파수 영역으로
    flimx = np.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))  # 전달함수가 엘리어싱 오류를 일으키지 않는 주파수의 범위를 나타낸 것. 
    flimy = np.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))  
    mask = np.zeros((nu, nv)) 
    mask = tf.cast(mask, dtype=tf.complex64) 
    mask = (np.abs(X) < flimx) & (np.abs(Y) < flimy) # 여기서 band limit하는 듯. X와 Y는 u를 뜻함. 
    mask = set_amplitude(h, mask) # phase와 amplitude로 변환 후 새로운 field인 mask로 계산.(cos,sin,1j 사용)
    field = tf.cast(field,dtype=tf.complex64)
    field = field[:,:,:,0]
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # 이미지 영역이 주파수 영역으로
    U2 = mask*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2)) # 다시 이미지 영역으로
    # recon = tf.math.abs(result)
    # recon = tf.cast(recon,tf.float32)

    return result

def band_limited_angular_spectrum2(field, k, distance, dx, wavelength, sampling_factor):

    nv, nu = field.shape[1],field.shape[2] # (756,1344)
    x = np.linspace(-nu+sampling_factor/2*dx, nu-sampling_factor/2*dx, nu) # x = (1344,)
    y = np.linspace(-nv+sampling_factor/2*dx, nv-sampling_factor/2*dx, nv) # y = (756,)
    X, Y = np.meshgrid(x, y) # X, Y 모두 (756, 1344)
    Z = X**2+Y**2 # 원형 모양 
    h = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h = np.fft.fft2(np.fft.fftshift(h))*dx**2 # 이미지 영역이 주파수 영역으로
    flimx = np.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))  # 전달함수가 엘리어싱 오류를 일으키지 않는 주파수의 범위를 나타낸 것. 
    flimy = np.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))  
    mask = np.zeros((nu, nv)) 
    mask = tf.cast(mask, dtype=tf.complex64) 
    mask = (np.abs(X) < flimx) & (np.abs(Y) < flimy) # 여기서 band limit하는 듯. X와 Y는 u를 뜻함. 
    mask = set_amplitude(h, mask) # phase와 amplitude로 변환 후 새로운 field인 mask로 계산.(cos,sin,1j 사용)
    field = tf.cast(field,dtype=tf.complex64)
    field = field[:,:,:,0]
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # 이미지 영역이 주파수 영역으로
    U2 = mask*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2)) # 다시 이미지 영역으로
    # recon = tf.math.abs(result)
    recon = tf.math.abs(result)/tf.reduce_max(tf.math.abs(result))
    # recon = tf.cast(recon,tf.float32)

    return recon

def band_limited_angular_spectrum3(field, k, distance, dx, wavelength, sampling_factor):

    nv, nu = field.shape[1],field.shape[2] # (756,1344)
    x = np.linspace(-(nu*sampling_factor)/2*dx, (nu*sampling_factor)/2*dx, nu) # x = (1344,) 
    y = np.linspace(-(nv*sampling_factor)/2*dx, (nv*sampling_factor)/2*dx, nv) # y = (756,)
    X, Y = np.meshgrid(x, y) # X, Y 모두 (756, 1344)
    Z = X**2+Y**2 # 원형 모양 
    h = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h = np.fft.fft2(np.fft.fftshift(h))*dx**2 # 이미지 영역이 주파수 영역으로
    flimx = np.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))  # 전달함수가 엘리어싱 오류를 일으키지 않는 주파수의 범위를 나타낸 것. 
    flimy = np.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))  
    mask = np.zeros((nu, nv)) 
    mask = tf.cast(mask, dtype=tf.complex64) 
    mask = (np.abs(X) < flimx) & (np.abs(Y) < flimy) # 여기서 band limit하는 듯. X와 Y는 u를 뜻함. 
    mask = set_amplitude(h, mask) # phase와 amplitude로 변환 후 새로운 field인 mask로 계산.(cos,sin,1j 사용)
    field = tf.cast(field,dtype=tf.complex64)
    field = field[:,:,:,0] # 756,1344
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # 이미지 영역이 주파수 영역으로
    U2 = mask*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2)) # 다시 이미지 영역으로
    # recon = tf.math.abs(result)
    # recon = tf.cast(recon,tf.float32)

    return result


def band_limited_angular_spectrum4(field, k, distance, dx, wavelength, sampling_factor):

    nv, nu = field.shape[1],field.shape[2] # (756,1344)
    x = np.linspace(-(nu*sampling_factor)/2*dx, (nu*sampling_factor)/2*dx, nu) # x = (1344,)  
    y = np.linspace(-(nv*sampling_factor)/2*dx, (nv*sampling_factor)/2*dx, nv) # y = (756,)
    X, Y = np.meshgrid(x, y) # X, Y 모두 (756, 1344)
    Z = X**2+Y**2 # 원형 모양 
    h = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h = np.fft.fft2(np.fft.fftshift(h))*dx**2 # 이미지 영역이 주파수 영역으로
    flimx = np.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))  # 전달함수가 엘리어싱 오류를 일으키지 않는 주파수의 범위를 나타낸 것. 
    flimy = np.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))  
    mask = np.zeros((nu, nv)) 
    mask = tf.cast(mask, dtype=tf.complex64) 
    mask = (np.abs(X) < flimx) & (np.abs(Y) < flimy) 
    mask = set_amplitude(h, mask) # phase와 amplitude로 변환 후 새로운 field인 mask로 계산.(cos,sin,1j 사용)
    field = tf.cast(field,dtype=tf.complex64)
    field = field[:,:,:,0]
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # 이미지 영역이 주파수 영역으로
    U2 = mask*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2)) # 다시 이미지 영역으로
    recon = tf.math.abs(result)
    # recon = tf.math.abs(result)/tf.reduce_max(tf.math.abs(result))
    # recon = tf.cast(recon,tf.float32)

    return recon

def band_limited_angular_spectrum5(field, k, distance, dx, wavelength, sampling_factor):

    nv, nu = field.shape[1],field.shape[2] # (756,1344)
    x = np.linspace(-nu/2*dx, nu/2*dx, nu) # x = (1344,)  
    y = np.linspace(-nv/2*dx, nv/2*dx, nv) # y = (756,)
    X, Y = np.meshgrid(x, y) # X, Y 모두 (756, 1344)
    Z = X**2+Y**2 # 원형 모양 
    h = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h = np.fft.fft2(np.fft.fftshift(h))*dx**2 # 이미지 영역이 주파수 영역으로
    flimx = np.ceil(1/(((2*distance*(1./(nu*sampling_factor)))**2+1)**0.5*wavelength))  # 전달함수가 엘리어싱 오류를 일으키지 않는 주파수의 범위를 나타낸 것. 
    flimy = np.ceil(1/(((2*distance*(1./(nv*sampling_factor)))**2+1)**0.5*wavelength))  
    mask = np.zeros((nu, nv)) 
    mask = tf.cast(mask, dtype=tf.complex64) 
    mask = (np.abs(X) < flimx) & (np.abs(Y) < flimy) # 여기서 band limit하는 듯. X와 Y는 u를 뜻함. 
    mask = set_amplitude(h, mask) # phase와 amplitude로 변환 후 새로운 field인 mask로 계산.(cos,sin,1j 사용)
    field = tf.cast(field,dtype=tf.complex64)
    field = field[:,:,:,0] # 756,1344
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # 이미지 영역이 주파수 영역으로
    U2 = mask*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2)) # 다시 이미지 영역으로
    # recon = tf.math.abs(result)
    # recon = tf.cast(recon,tf.float32)

    return result


def band_limited_angular_spectrum6(field, k, distance, dx, wavelength, sampling_factor):

    nv, nu = field.shape[1],field.shape[2] # (756,1344)
    x = np.linspace(-nu/2*dx, nu/2*dx, nu) # x = (1344,)  => 여기서의 샘플링 팩터의 개념을 정확히 파악해야함. 
    y = np.linspace(-nv/2*dx, nv/2*dx, nv) # y = (756,)
    X, Y = np.meshgrid(x, y) # X, Y 모두 (756, 1344)
    Z = X**2+Y**2 # 원형 모양 
    h = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h = np.fft.fft2(np.fft.fftshift(h))*dx**2 # 이미지 영역이 주파수 영역으로
    flimx = np.ceil(1/(((2*distance*(1./(nu*sampling_factor)))**2+1)**0.5*wavelength))  # 전달함수가 엘리어싱 오류를 일으키지 않는 주파수의 범위를 나타낸 것. 
    flimy = np.ceil(1/(((2*distance*(1./(nv*sampling_factor)))**2+1)**0.5*wavelength))  
    mask = np.zeros((nu, nv)) 
    mask = tf.cast(mask, dtype=tf.complex64) 
    mask = (np.abs(X) < flimx) & (np.abs(Y) < flimy) # 여기서 band limit하는 듯. X와 Y는 u를 뜻함. 
    mask = set_amplitude(h, mask) # phase와 amplitude로 변환 후 새로운 field인 mask로 계산.(cos,sin,1j 사용)
    field = tf.cast(field,dtype=tf.complex64)
    field = field[:,:,:,0]
    U1 = tf.signal.fft2d(tf.signal.fftshift(field)) # 이미지 영역이 주파수 영역으로
    U2 = mask*U1
    result = tf.signal.ifftshift(tf.signal.ifft2d(U2)) # 다시 이미지 영역으로
    recon = tf.math.abs(result)
    # recon = tf.math.abs(result)/tf.reduce_max(tf.math.abs(result))
    # recon = tf.cast(recon,tf.float32)

    return recon


def calculate_amplitude(field):

    amplitude = np.abs(field)
    return amplitude

def calculate_phase(field, deg=False):

    phase = np.angle(field)
    if deg == True:
        phase *= 180./np.pi
    return phase

def set_amplitude(field, amplitude): # h, mask

    amplitude = calculate_amplitude(amplitude)
    phase = calculate_phase(field)
    new_field = amplitude*np.cos(phase)+1j*amplitude*np.sin(phase) # 오일러 공식(phase는 angle이니) e의 iangle승
    return new_field

def propagation_ASM(field, feature_size, wavelength, z, return_H=False, precomputed_H=None, dtype=torch.float32):
    if precomputed_H is None: # precomputed_H가 없는 경우 H를 공식으로 직접 구한 후에 u_out을 계산
        # field_resolution = u_in.size()  # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        num_y, num_x = field.shape[0],  field.shape[1]  # number of pixels
        dy, dx = feature_size  # sampling inteval size
        y, x = (dy * float(num_y), dx * float(num_x))  # size of the field

        # frequency coordinates sampling  x = np.linspace(-nu/2*dx, nu/2*dx, nu)
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y) # 0.5를 더하고 빼는게 추가된 듯
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance) 전송함수
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))
        HH = torch.tensor(HH, dtype=dtype).to(field.device)
        HH = torch.reshape(HH, (1, 1, *HH.size())) #  HH는 (1,1,HH의 사이즈)
        H_exp = torch.mul(HH, z).to(field.device)   # multiply by distance  HH의 각 요소에 z가 곱해짐

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype).to(u_in.device)

        # get real/img components
        H_real, H_imag = polar_to_rect(H_filter.to(field.device), H_exp) # 밴드 리미트된 필터로 real과 imaginary 만듦
        H = torch.stack((H_real, H_imag), -1) # 새로운 차원으로 주어진 텐서들을 붙인다. 
        H = torch.fft.ifftshift(H)
        H = torch.view_as_complex(H) # ex. tensor([1.6116, -0.5772]) => tensor([(1.6116-0.5772j)])
    else:
        H = precomputed_H   # precomputed_H가 있는 경우는 H를 precomputed_H로 대입해서 밑에 u_out을 계산

    if return_H:
        return H
    else:
        U1 = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(field)))
        U2 = H * U1
        u_out = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(U2)))
        return u_out

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def spherical(distace,wavelength):
    
    nv = 756
    nu = 1344

    pix = 8.5e-6
    nv_pix = nu * pix #LN
    nu_pix = nv * pix #LM
    L0 = wavelength * distace / pix
    n = tf.linspace(0, nv - 1, nv)
    m = tf.linspace(0, nu - 1, nu)
    x0 = -L0 / 2 + L0 / nu * m
    y0 = -L0 / 2 + L0 / nv * n
    x_img, y_img = tf.meshgrid(x0, y0)
    x = -nu_pix / 2 + nu_pix / nu * m
    y = -nv_pix / 2 + nv_pix / nv * n
    x_slm, y_slm = tf.meshgrid(x, y)
    L_img = (x_img ** 2 + y_img ** 2) / distace # 이게 공간 주파수에 거리 나눈것
    L_slm = (x_slm ** 2 + y_slm ** 2) / distace  # slm의 공간 주파수에 거리 나눈것
    theta_img = (np.pi / wavelength * L_img) # lens fn
    theta_slm = (np.pi / wavelength * L_slm) # lens fn
    theta_slm = tf.cast(theta_slm,dtype=tf.float32)

    return theta_slm

def spherical_phase(distance,wavelength,pixel_size):
        nv, nu = 756,1344
        x = np.linspace(-nu/2*pixel_size, nu/2*pixel_size, nu)
        y = np.linspace(-nv/2*pixel_size, nv/2*pixel_size, nv)
        X, Y = np.meshgrid(x, y)
        Z = X**2+Y**2
        # spherical = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/(2*distance)*Z) # complex / 오류 unsupported operand type(s) for *: 'complex' and 'CPUDispatcher'
        # spherical = np.exp(1j*k*distance*np.sqrt(1-(wavelength*X)**2-(wavelength*Y)**2))
        spherical = np.exp(np.remainder(np.pi / distance * (Z) / wavelength, 2 * np.pi)) # exp로 complex되지는 않음. 
        spherical = tf.cast(spherical,dtype=tf.complex64) # 3번 할때만 특별히 
        spherical = tf.signal.fft2d(tf.signal.fftshift(spherical,axes=[0,1]))*pixel_size**2 # complex
        spherical = spherical[np.newaxis,:,:] # (1,756,1344)
        spherical = tf.cast(spherical,dtype=tf.float64)

        return spherical



# 방법 7
def prop_to_slm_phase(amp,lens):
    
    random = np.random.random((1024,1024))
    # amp180 = cv2.rotate(amp, cv2.ROTATE_180) # np array만 받을 수 있음.
    # reversed_amp = amp[::-1]
    # amp = amp[:,:,:,0] # network에 넣기전에 쓴다면 없애고 넣은 후에 쓴다면 활성화한다.
    # amp = tf.expand_dims(amp,axis=-1)
    amp = tf.cast(amp,tf.complex64)
    complex_field = amp*tf.exp(1j*amp) # (N,1024,1024)
    complex_field_fftshift = tf.signal.fftshift(complex_field,axes=[1,2])
    complex_field_fft =  tf.signal.fft2d(complex_field_fftshift) # 방법 2-> 여기서 [400,1024,1024] OOM 오류 뜸.
    lens = tf.cast(lens,tf.complex64)
    complex_field_fft = complex_field_fft*lens 
    slm_phase = tf.math.angle(complex_field_fft) # prop(lens)없이 phase를 만든 상태
    # slm_phase = tf.expand_dims(slm_phase,axis=-1)
    # slm_phase = tf.cast(slm_phase,tf.complex64)

    return slm_phase

def prop_to_slm_phase2(amp,lens):
    
    random = np.random.random((1024,1024))
    # amp180 = cv2.rotate(amp, cv2.ROTATE_180) # np array만 받을 수 있음.
    # reversed_amp = amp[::-1]
    amp = amp[:,:,:,0]
    amp = tf.cast(amp,tf.complex64)
    complex_field = amp*tf.exp(1j*random) #(756,1344)
    # complex_field_fft = tf.signal.fft2d(tf.signal.fftshift(complex_field,axes=[0,1]))
    complex_field_ifftshift = tf.signal.ifftshift(complex_field,axes=[1,2])
    complex_field_ifft =  tf.signal.ifft2d(complex_field_ifftshift)
    lens = tf.cast(lens,tf.complex64)
    complex_field_ifft = complex_field_ifft*lens # cf엔 ifft lens엔 fft 되어있는 상태
    slm_phase = tf.math.angle(complex_field_ifft) # prop(lens)없이 phase를 만든 상태
    # slm_phase = tf.cast(slm_phase,tf.complex64)

    return slm_phase



def recon_prop(slm_phase,lens):
    slm_phase = tf.cast(slm_phase,tf.complex64)
    slm_phase = slm_phase-tf.reduce_mean(slm_phase)
    prop_field = slm_phase*lens # phase는 fftshift한 lens랑 먼저 곱해진 후에 fftshift,fft,ifftshift가 되어진다.
    recon = tf.signal.fftshift(prop_field,axes=[1,2])
    recon = tf.signal.fft2d(recon)
    recon = tf.signal.ifftshift(recon,axes=[1,2])
    recon = tf.abs(recon)
    recon = tf.cast(recon,tf.float64)
    # recon = tf.math.abs(recon)
    
    return recon

def recon_prop2(slm_phase,lens):

    slm_phase = tf.cast(slm_phase,tf.complex64)
    slm_phase = slm_phase-tf.reduce_mean(slm_phase)
    prop_field = slm_phase*lens # phase는 fftshift한 lens랑 먼저 곱해진 후에 fftshift,fft,ifftshift가 되어진다.
    recon = tf.signal.ifftshift(prop_field,axes=[1,2])
    recon = tf.signal.ifft2d(recon)
    recon = tf.signal.ifftshift(recon,axes=[1,2])
    # recon = tf.abs(recon)/tf.reduce_max(tf.abs(recon))
    recon = tf.math.abs(recon)
    
    return recon

def prop_to_slm_phase5(amp,lens): # (N,1024,1024,1024)의 문제점을 해결하기 위해서 생성.
    
    random = np.random.random((1024,1024))
    amp = amp[:,:,:,0]
    # amp = tf.expand_dims(amp[:,:,:,0],axis=-1)
    amp = tf.cast(amp,tf.complex128)
    
    complex_field = amp*tf.exp(1j*random) #(1024,1024)
    complex_field_fftshift = tf.signal.fftshift(complex_field,axes=[1,2])
    complex_field_fft =  tf.signal.fft2d(complex_field_fftshift)
    lens = tf.cast(lens,tf.complex128)
    complex_field_fft = complex_field_fft*lens 
    slm_phase = tf.math.angle(complex_field_fft) # prop(lens)없이 phase를 만든 상태
    slm_phase = tf.cast(slm_phase,tf.complex64)
    

    return slm_phase
