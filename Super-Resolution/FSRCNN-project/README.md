# 소규모 데이터셋에서의 FSRCNN 성능을 높이기 위한 프로젝트(FSRCNN-s)

이 프로젝트는  "Accelerating the Super-Resolution Convolutional Neural Network" 논문을 구현한 프로젝트이다.

![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/6b5f1d46-98e6-49d5-a14b-c5f31875b009)

# 프로젝트 목표
1. 소규모 데이터셋 기준 기존의 FSRCNN보다 더 나은 복원 퀄리티
2. 길지 않은 훈련 시간

# 기존 논문 구현 결과
![image.jpg1](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e7468b8-3860-4ce1-8d83-f2153effd04e)|![low_resolution_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/f692903b-90e8-4bc8-99e0-8a9ec677cbb1)|![bic_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/2aa18f3b-b703-4b78-86f1-3bcb5b7ab6cc)|![original_fsrcnn_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/12f0762b-c6bb-478d-8732-ce0fa2f11443)
--- | --- | --- | --- |
|Ground truth|Low resolution|Bicubic|Original FSRCNN|


|모델|PSNR|SSIM|Training time|
|---|---|---|---|
|Bicubic|*26.9*|0.91|-|
|Original FSRCNN|*24.98*|0.87|56m 47s|

# 수정한 부분(FSRCNN-s)
- SGD optimizer 대신 Adam optimizer를 사용하였다. 
- Adam은 소규모 데이터셋에서도 좋은 성능을 발휘할 수 있기 때문에 소규모 데이터셋을 선택하였다. 
- 소규모 데이터셋을 선택하였기 때문에 flip을 활용하여 데이터셋 증강을 하였음. 
- 수정한 모델을 FSRCNN-s으로 이름을 정하였다.(소규모 데이터셋을 위한 모델이기 때문에 small의 약자 s를 붙임)

# 수정 후 FSRCNN-s의 결과
![butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e7468b8-3860-4ce1-8d83-f2153effd04e)|![low_resolution_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/da2360fc-f23d-479c-9c70-45230875f367)|![bic_butterfly](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/7b5eff26-1e1a-4bba-82d6-9b7c2e906962)|![fsrcnn_butterfly2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/8e95cc29-f20f-4c09-86ae-0ad9edee40d9)
--- | --- | --- | --- |
|Ground truth|Low resolution|Bicubic|FSRCNN-s

|모델|PSNR|SSIM|Training time|
|---|---|---|---|
|Bicubic|*26.9*|0.91|-|
|FSRCNN-s|*27.8*|0.92|1h 43s|

- 기존의 FSRCNN보다 PSNR과 SSIM이 모두 상승하였음을 확인할 수 있고, 육안으로도 더 또렷하고 엣지를 더 명확하게 보여주는 결과를 나타냈음을 확인하였다.
- SGD optimizer에서 Adam optimzer로 바꿈으로써, 소규모 데이터셋을 사용함에도 불구하고 더 좋은 성능을 발휘하였다.

### 사용 버전 
- tensorflow 2.4
- python 3.6.5

### 사용한 소규모 이미지 데이터셋

- yang91 dataset


