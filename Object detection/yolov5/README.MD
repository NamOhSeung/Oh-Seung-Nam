# Yolov5를 이용한 축구 경기에서 선수들의 유니폼을 통한 팀 검출

## 프로젝트 목적
- 이 프로젝트의 목표는 YOLO(You Only Look Once) 객체 감지 모델을 활용하여 축구 경기 중 각 팀의 유니폼과 공을 소유한 선수를 실시간으로 탐지하고 추적하는 것이다. 
- 이를 통해 각 팀의 선수 위치 파악 뿐만이 아니라 추후에 전술 분석, 방송 중계에 활용될 수 있는 데이터를 제공하는 시스템을 개발하도록 한다.

## 데이터 수집 및 준비
  
|![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/de3e19c3-871f-4e9f-902c-c86d8e93e08b)|![image](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/f174a807-6606-4515-a998-f4f1c8cf1c71)|
|:---:|:---:|
|image|bounding box|


- 잉글랜드의 축구 클럽인 맨체스터 유나이티드(빨간색 유니폼)와 맨체스터 시티(하늘색 유니폼) 경기에서의 유니폼 캡처와 구글에서의 유니폼 사진을 수집하여 데이터셋을 구성하였다.
- Train, validation을 각각 9:1 비율로 구성하였다.(총 150개의 이미지)
- Label은 man utd(맨체스터 유나이티드), man city(맨체스터 시티)로 구성하였다.
- 데이터 라벨링 프로그램(https://github.com/tzutalin/labelImg) 을 이용하여 각 이미지에 유니폼별로 바운딩 박스를 그려 라벨링을 생성하였다.

## 모델 학습
Yolov5로 학습을 진행하였으며, epoch은 100으로 처리하였다.
클래스별 mAP 측정 결과 최대 0.956을 도출 했습니다.

## 결과

|![football_colab (1) (2)](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/0314e39e-4059-495f-9248-d9b65c059093)|
|:---:|
|result video|

- 맨체스터 유나이티드와 맨체스터 시티 경기의 동영상을 일부 잘라내어 test image로 사용을 하였다.
- 유니폼 별로 클래스를 구분한 결과, 각 팀의 선수들의 팀이 잘 검출되었다.
- 아쉬운 점은 클래스를 구분하지 않았던 심판도 가끔씩 선수로 검출이 되는 모습이 있었다.
