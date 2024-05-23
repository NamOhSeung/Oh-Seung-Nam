# EDSR
![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/74a7b150-ed6e-4887-bc75-933ee03a0b7f)

tensorflow와 keras로 "Enhanced Deep Residual Networks for Single Image Super-Resolution" from CVPRW 2017, 2nd NTIRE 논문을 구현하였다. 
링크 : https://arxiv.org/abs/1707.02921

# EDSR 결과
![prediction 비교](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/83272d37-171b-41b2-8024-60869e297a6a)

|모델|PSNR|SSIM|
|---|---|---|
|Bicubic|31.2|0.975|
|EDSR|32.3|0.987|

![bicubic_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/3d46e8bf-d5df-40b1-920d-b13ba3f017fa)|![edsr_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/ac4682b7-f832-49c1-b8b1-cfd53e0b14fd)
--- | --- |
|Bicubic(1024*1024)|EDSR(1024*1024)|

- 저해상도의 이미지를 입력 이미지로 선정하고, bicubic 보간법을 통해 저해상도 이미지를 고해상도 이미지로 구현 후, EDSR 모델을 통해 저해상도의 이미지를 고해상도 이미지와 같은 해상도의 이미지로 구현해 보았다.

- 이를 통해 bicubic으로 고해상도로 변환한 것보다 EDSR 모델을 통해 구현한 고해상도의 이미지가 눈으로 봤을 때 엣지를 더 잘 잡아내고, 세밀하게 구현이 되는 것을 확인할 수 있다.
또한 수치적으로도 PSNR과 SSIM이 더욱 높게 나오는 것을 확인하였다.

### 실행
- main.ipynb

### 사용 버전
- tensorflow 2.8
- tensorflow-gpu 2.8

### 참고
Image Super Resolution Using EDSR and SRGAN : https://github.com/IMvision12/Image-Super-Resolution/tree/main
