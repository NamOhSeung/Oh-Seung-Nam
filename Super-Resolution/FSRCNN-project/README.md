# FSRCNN
이 프로젝트는  "Accelerating the Super-Resolution Convolutional Neural Network" 논문을 구현한 프로젝트이다.

![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/6b5f1d46-98e6-49d5-a14b-c5f31875b009)

# 프로젝트 목표
기존의 모델과 파라미터의 수정을 통해 기존의 모델을 통해 나온 결과보다 더 좋은 결과를 만들어내는 것이 목표이다.

# 기존 논문 구현 결과
![image.jpg1](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e7468b8-3860-4ce1-8d83-f2153effd04e)|![low_resolution_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/f692903b-90e8-4bc8-99e0-8a9ec677cbb1)|![bic_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/2aa18f3b-b703-4b78-86f1-3bcb5b7ab6cc)|![original_fsrcnn_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/12f0762b-c6bb-478d-8732-ce0fa2f11443)
--- | --- | --- | --- |
|Ground truth|Low resolution|Bicubic|Original FSRCNN|


|모델|PSNR|SSIM|
|---|---|---|
|Bicubic|*26.9*|0.91|
|Original FSRCNN|*24.98*|0.87|

# 수정한 부분(FSRCNN-s)
- 저자는 d=32, s=5, m=1을 사용하였지만, 여기서는 d=56, s=12, m=4을 사용하여 더 나은 복원 퀄리티를 생성하였다.
- SGD 대신 ADAM optimizer를 사용하였다. 연산량이 더 늘어나는데 비해, step size뿐만이 아니라 descent direction까지 조절이 되어 더 좋은 복원 퀄리티 결과를 만들었다.
- 수정한 모델을 FSRCNN-s으로 이름을 정하였다.(소규모 데이터셋을 위한 모델이기 때문에 small의 약자 s를 붙임)

# 수정 후 FSRCNN-n의 결과
![butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e7468b8-3860-4ce1-8d83-f2153effd04e)|![low_resolution_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/da2360fc-f23d-479c-9c70-45230875f367)|![bic_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/7b5eff26-1e1a-4bba-82d6-9b7c2e906962)|![fsrcnn_butterfly2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e95cc29-f20f-4c09-86ae-0ad9edee40d9)
--- | --- | --- | --- |
|Ground truth|Low resolution|Bicubic|FSRCNN-n

|모델|PSNR|SSIM|
|---|---|---|
|Bicubic|*26.9*|0.91|
|FSRCNN-n|*27.8*|0.92|

- 기존의 FSRCNN보다 PSNR과 SSIM이 모두 상승하였음을 확인할 수 있고, 육안으로도 더 또렷하고 엣지를 더 명확하게 보여주는 결과를 나타냈음을 확인하였다.

### 사용 버전 
- tensorflow 2.4
- python 3.6.5

### 이미지 데이터셋

- yang91 dataset

### 참고
https://github.com/OlgaChernytska/Super-Resolution-with-FSRCNN?tab=readme-ov-file


