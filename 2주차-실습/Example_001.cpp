//detect face+eyes by Haar-like Feature-based Cascade Classifiers
//#pragma warning(disable:4996)

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

//Functions
void detectAndDisplay(Mat frame);

//Global Vars
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;


int main()
{
	//local Vars
	String face_cascade_name = "C:/opencv-2-4-13-6/data/haarcascades/haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "C:/opencv-2-4-13-6/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
	int camera_device = 0;

	//Haar cascade 로드
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return -1;
	};

	//Cam 로드
	VideoCapture capture;
	if (!capture.open(camera_device)) {
		cout << "--(!)Error open camera device\n";
		return -1;
	}

	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
//			cout << "--(!) No captured frame -- Break!\n";
//			break;

			//일시적인 오류일 수도 있으므로 continue 합니다.
			cout << "--(!) No captured frame --\n";
			continue;

		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

		if (!waitKey(10))
		{
			break; // escape
		}
	}
	return 0;
}

void detectAndDisplay(Mat frame_origin)
{
	Mat frame;
	cvtColor(frame_origin, frame, COLOR_BGR2GRAY);
	//frame의 Histogram을 정규화
	equalizeHist(frame, frame);

	//face 검출
	vector<Rect> faces;
	cout << "detecting faces...\n" << endl;
	face_cascade.detectMultiScale(frame, faces);	//1147ms
	cout << "fin!" << endl;

	//각 얼굴마다
	for (size_t i = 0; i < faces.size(); i++)
	{
		//얼굴의 중점 (x,y) 좌표는 좌상단좌표+(길이/2)
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//타원 그리기(원본 프레임, 중점, main axes/2, 회전각(deg), 호(arc)의 시작 / 끝 각(deg), 선 색, 두께)
		ellipse(frame_origin, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		
		//Region of Interest, mat 안에 roi를 넣습니다.
		Mat faceROI = frame(faces[i]);

		//눈 검출
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);	//281ms

		//각 눈마다
		for (size_t j = 0; j < eyes.size(); j++)
		{
			//눈의 중점, 반지름
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame_origin, eye_center, radius, Scalar(255, 0, 0), 4);
		}
	}

	//result
	imshow("얼굴 검출", frame_origin);
}
