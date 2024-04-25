# 논문제목

BL-ASM에서 U-net 기반 위상 홀로그램의 스펙클 노이즈 감소와 이미지 품질 향상

# 논문 요약

Band-limited angular spectrum method (BL-ASM)는 공간주파수 제어의 문제로 aliasing 오류가 발생한다. 본 논문에서는 위상 홀로그램에 대한 표본화 간격 조정 기법과 딥 러닝 기반의 U-net 모델을 사용한 스펙클 노이즈 감소 및 이미지 품질 향상 기법을 제안하였다. 
제안한 기법에서는 넓은 전파 범위에서 aliasing 오류를 제거할 수 있도록 먼저 샘플링 팩터를 계산하여 표본화 간격 조절에 의한 공간주파수를 제어함으로써 스펙클 노이즈를 감소시킨다. 
그 후 딥 러닝 모델을 적용한 위상 홀로그램을 학습시켜 복원 이미지의 품질을 향상시킨다. 다양한 샘플 이미지에 대한 S/W 시뮬레이션에서 기존의 BL-ASM과의 peak signal-to-noise ratio (PSNR), structural similarity index measure(SSIM)을 비교할 때 각각 평균 5%, 0.14% 정도 비율이 향상됨을 확인하였다.

# 순서도

![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/b981cd5b-b64b-4a7a-b8e9-abffa73e530e)

위 그림은 본 논문에서 제안한 샘플링 팩터를 활용해 공간 주파수를 제어한 뒤, 딥 러닝(Unet)을 적용하는 방법에 대한 순서도이다.

# 모델 설정

![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/3bb66278-8e9a-417e-a549-60b9fd18cc2f)
![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/89f97ece-b435-4f90-b662-4a4006ede5ae)

위 그림과 같이 모델을 설정하였다.
기존의 U-Net과의 다른점이라 한다면, 처음 필터와 최종 필터의 수가 비대칭으로 이루어져있다는 것이다.
이는, 기존과 다르게 이와 같이 설정하였을 때, PSNR의 값이 최소 0.1에서 최대 0.4까지 상승하는 것을 확인하였기 때문이다.

# 결과

### 각 이미지별 복원 홀로그램 이미지 
![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/0ff84418-68eb-4cae-be34-63a4d587cc0d)

복원 이미지는 전파거리 0.5–1.5 m 사이에서 생성한 각 방법마다의 복원 이미지 중 PSNR이 가장 높은 것을 선정하였다. 이는 가장 높은 이미지 품질을 가진 복원 이미지끼리 비교하여 정량적 평가를 가능케 하기 위해서이다.

---
### 한 이미지를 통한 스펙클 노이즈와 이미지 퀄리티 비교
![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/a0232a3b-54bb-48e0-84ad-402e2303b7d6)
![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/a81f9524-ce0e-496c-b00f-3ca85f4e4f42)


그림 (a)(왼쪽 위)는 원본 이미지, 그림 (b)(오른쪽 위)는 BL-ASM으로 위상 홀로그램을 복원한 이미지, 그림 (c)(왼쪽아래)는 샘플링 팩터를 조정하여 공간 주파수를 제어한 BL-ASM으로 복원한 이미지, 그리고 그림 (d)(오른쪽아래)는 제안한 방법인 샘플링 팩터를 조정하여 공간 주파수를 제어한 BL-ASM을 가지고 딥 러닝을 적용하여 생성한 복원 이미지

---

![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/7cfa8c93-07fd-4124-9891-cd459c78901f)

스펙클 노이즈의 정량적 평가를 위해 그림 (b)–(d)에서 세 번째 구역의 스펙클 노이즈를 pixel 값으로 표현한 그래프이다

---

![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/b2d46959-02b9-4f7e-9c35-508aa26afe18)

PSNR 비교

---

![image](https://github.com/NamOhSeung/Speckle-Noise-Reduction-and-Image-Quality-Improvement-in-U-net-based-Phase-Holograms-in-BL-ASM/assets/98510923/047dd063-1ab4-4bcb-aee3-428465d19555)

SSIM 비교

# 필요 버전
- python 3.6.5
- tensorflow 2.4
- numpy 1.19
- scipy 1.5.4
- matplotlib 3.3.4



# 논문 링크
《연구논문》 Korean Journal of Optics and Photonics, Vol. 34, No. 5, October 2023, pp. 192-201
DOI: https://doi.org/10.3807/KJOP.2023.34.5.192


