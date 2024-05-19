# AUTOENCODER를 활용한 노이즈 제거 
![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/cc920e0f-a05a-4dcb-af94-92f5704ce305)

이미지 출처 : https://excelsior-cjh.tistory.com/187

## 데이터셋
텐서플로우 내장 데이터셋인 MNist 사용

## 결과

- 테스트 이미지

![test 이미지 모음1](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/e83f3635-77b1-42f7-80d2-63b053ccf801)
![test 이미지 모음2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/a134a147-abe4-45e7-ba05-bc1d4fb8ef7b)

- 노이즈 이미지

![noise_image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/15d2f8d3-dcfe-4ba9-b784-d115aa0f0f24)
![noise_image2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/fda3323d-e81c-4332-9930-1397555c21fa)

- 노이즈 제거한 이미지

![prediction 비교_conv layer 모델](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/13114dbe-cfba-4f9f-84f9-69019cd5bce3)
![prediction 비교_conv layer 모델2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/fadb79d7-e7e1-46d6-a147-71db17e3ec9f)

- 위 이미지는 테스트 이미지, 가우시안 노이즈를 섞은 테스트 이미지, 오토인코더 모델을 적용하여 노이즈를 제거한 이미지이다.
- 모델은 Autoencoder를 활용하였다.
- Autoencoder 모델을 통해 가우시안 노이즈가 거의 제거된 모습을 확인할 수 있다. 

# 사용 버전
- tensorflow 2.8
- tensorflow-gpu 2.8

# 실행
- main.py
