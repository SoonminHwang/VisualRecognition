# xcode build 환경설정

## home brew 없을시 커맨드창에 입력


> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

## opencv 2버전 설치

> brew install opencv@2

> brew install pkg-config

> pkg-config --cflags --libs opencv



출력이 다음과 같이 나오는데 출력 결과를 복사 해둡시다~

```
-I/usr/local/Cellar/opencv@2/2.4.13.7_2/include/opencv -I/usr/local/Cellar/opencv@2/2.4.13.7_2/include -L/usr/local/Cellar/opencv@2/2.4.13.7_2/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
gim-yeonjun-ui-MacBookPro:Desktop iwin247$ cd /usr/local/Cellar/opencv/2.4.13.7_2/lib
```

* xcode를 열어 새로운 프로젝트 만들기 -> mac os -> command line 생성

[여기 들어가서 Set Header Search Paths 부터 따라하시면되요](https://medium.com/@jaskaranvirdi/setting-up-opencv-and-c-development-environment-in-xcode-b6027728003#f6b6)

other link 설정은 아까 출력 -I뭐시기 나온거 입력하시면되고

brew로 다운받으시면 opencv 풀더위치 이렇게 되있을꺼에요

/usr/local/Cellar/opencv/2.4.13.7_2/

[웹캠 안되시면 일로 들어가서 설정하시면되요~](https://stackoverflow.com/questions/53190412/accessing-webcam-in-xcode-with-opencv-c)
