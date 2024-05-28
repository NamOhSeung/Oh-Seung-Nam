# R-CNN 구현

|![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/55650025-3e75-4334-8bd9-e2818ab32a8c)|
|:---:|
|R-CNN 구조 이미지|



### 사용 버전
- python 3.8
- tensorflow 2.10


### 결과

- loss
  
![loss](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/566ac82e-db9b-4cc3-b39d-a650d18f658c)

정확도는 accuracy = 0.95, val accuracy = 0.0226로 나왔다.
  

- object detection image
  
|![output1](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/30243338-c511-4e63-84d1-7ba8da9a7b3d)|![output2](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/f3fe5174-aec6-4653-8e58-ca46f7bd135e)|![output3](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/9a95455a-6b3d-4a3f-a72b-c5b846fb246e)|![output4](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/757e20c4-5b95-4d5f-b886-e1fbff46d4ad)|
|:---:|:---:|:---:|:---:|
|output1|output2|output3|output4|

- 고찰

이미지 결과를 보았을 때, 비행기를 꽤 detection한 것을 확인 할 수 있었다.
bounding box와 selective search를 활용한 딥러닝을 처음으로 구현해보았는데, 이미지 화질 개선만 하다가 객체 인식의 데이터셋을 사용해서
생소한 부분도 있었지만 앞으로는 객체 인식에서의 데이터셋도 막히지 않고 유연하게 잘 사용할 수 있을 것 같다.

학습을 하면서 R-CNN은 역시 region proposal을 수천개 뽑은 후 각 영역마다 CNN을 수행해서 연산량이 커 수행 시간이 느리다는 것을 체감하였다.
그리고 학습에서의 모델이 총 3개나 있기 때문에 학습을 한 번에 할 수 없으며 각 region proposal에 대해 conv 연산을 공유하지 않기 때문에 end-to-end가 성립이 되지 않는다. 

이러한 문제점들을 해결한 Fast R-CNN을 그 다음으로 구현해볼 예정이다.


  




