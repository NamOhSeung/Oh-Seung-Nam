# FSRCNN
이 프로젝트는  "Accelerating the Super-Resolution Convolutional Neural Network" 논문을 구현한 프로젝트이다.

![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/6b5f1d46-98e6-49d5-a14b-c5f31875b009)

# 프로젝트 목표
기존의 모델과 파라미터의 수정을 통해 기존의 모델을 통해 나온 결과보다 더 좋은 결과를 만들어내는 것이 목표이다.

# 기존 논문 구현 결과

|모델|PSNR|SSIM|
|---|---|---|
|FSRCNN-n|*강조1*|테스트3|

# 수정한 부분(FSRCNN-n)
- 저자는 d=32, s=5, m=1을 사용하였지만, 여기서는 d=56, s=12, m=4을 사용하여 더 나은 복원 퀄리티를 생성하였다.
- SGD 대신 ADAM optimizer를 사용하였다. 연산량이 더 늘어나는데 비해, step size뿐만이 아니라 descent direction까지 조절이 되어 더 좋은 복원 퀄리티 결과를 만들었다.

# 수정 후 FSRCNN-n의 결과
![butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e7468b8-3860-4ce1-8d83-f2153effd04e)
<figcaption> Ground truth image </figcaption>


bicubic image

![fsrcnn_butterfly2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e95cc29-f20f-4c09-86ae-0ad9edee40d9)

FSRCNN-n image


|모델|PSNR|SSIM|
|---|---|---|
|FSRCNN-n|*강조1*|테스트3|

### 사용 버전 
- tensorflow 2.4
- python 3.6.5

### 참고
https://github.com/OlgaChernytska/Super-Resolution-with-FSRCNN?tab=readme-ov-file


