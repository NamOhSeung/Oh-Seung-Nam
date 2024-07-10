# 드라마 속 인물 얼굴을 인식하는 프로젝트

## 데이터셋 선정
- 드라마 '응답하라 1988'의 드라마 속 인물들을 캡처하여, 데이터셋을 총 300장 구성
- 데이터셋은 Train 240장, Valid 60장으로 Train, Valid를 8:2로 구성
- 클래스는 5명의 주인공인 'HYERI', 'Ryu Jun-yeol', 'Park Bo-gum', 'Lee Dong Hwi', 'Ko Kyung-pyo'로 구성

## 파라미터
- epoch은 200으로, optimizer는 SGD로 설정

## 결과

all과 각 클래스 별로 mAP와 mAP50-95를 정리하면,

|Class|mAP|mAP50-95|
|---|---|---|
|all|31.2|0.975|
|HYERI|32.3|0.987|
|Ryu Jun-yeol|32.3|0.987|
|Park Bo-gum|32.3|0.987|
|Lee Dong Hwi|32.3|0.987|
|Ko Kyung-pyo|32.3|0.987|

다음은 테스트 영상에 얼굴 인식이 잘 되었는지 확인한 결과이다.


|![전체인원](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/4d0aefe2-9d07-41d7-a8cb-07e94db1a175)|![혜리](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/a07d9c7e-3bfe-430d-8300-962ad25f44b5)|
|:---:|:---:|
|전체인원|혜리|

|![류준열](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/c174efff-e3b8-4cbc-82d9-b90def324b1f)|![박보검](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/55eacb48-863c-4d39-a9a3-c6e7bd60b207)
|:---:|:---:|
|류준열|박보검|

|![이동휘](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/4d95e74b-81d4-48ac-91e2-07b26c37ad75)|![고경표](https://github.com/NamOhSeung/Oh-Seung-Nam/assets/98510923/a55eda39-94c1-401f-b1b9-a43995e2321f)|
|:---:|:---:|
|이동휘|고경표|

드라마 속 인물들의 얼굴인식이 한 프레임도 빠짐 없이 잘 되어지는 것을 확인할 수 있다.
