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

bicubic으로 고해상도로 변환한 것보다 EDSR 모델을 통해 구현한 고해상도의 이미지가 더 세밀하게 구현이 되는 것을 확인하였다.

### 참고
Image Super Resolution Using EDSR and SRGAN : https://github.com/IMvision12/Image-Super-Resolution/tree/main
