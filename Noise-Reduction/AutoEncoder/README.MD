# AUTOENCODER를 활용한 노이즈 제거 
![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/cc920e0f-a05a-4dcb-af94-92f5704ce305)
이미지 출처 : https://excelsior-cjh.tistory.com/187

## 데이터셋
텐서플로우 내장 데이터셋인 MNist 사용

## 결과
![prediction 비교 1](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/792c8af4-44a0-4735-9609-1fd0d5a42494)
![prediction 비교 2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/4fe0b32c-2340-48c9-b550-e1aabe529bcb)

- 위 이미지는 테스트 이미지, 테스트 이미지에 노이즈를 섞은 이미지, 오토인코더 모델을 적용하여 노이즈를 제거한 이미지이다.
- 모델은 Autoencoder를 활용하였다.

# 사용 버전
- tensorflow 2.8
- tensorflow-gpu 2.8

# 실행
- main.ipynb
