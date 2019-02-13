#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
/////////////////////////////////////////////
//*****please setting the number of images*****
#define NUM 3
/////////////////////////////////////////////

Mat MakePano(Mat *imgArray, int num);

int main() {
	Mat result;
	Mat imgArray[NUM];

	//input images
	printf("input images...\n");
	for (int i = 0; i < NUM; i++)
	{
		imgArray[i] = imread("./images/" + std::to_string(i + 1) + ".jpg");
	}

	//Panormama stitching func start
	printf("Panormama stitching func start...\n");
	result = MakePano(imgArray, NUM);
	printf("func finished!\n");

	//shows image
	printf("shows image...\n");
	imshow("result", result);

	waitKey();
	return 0;
}

Mat MakePano(Mat *imgArray, int num)
{
	//panorama base
	printf("panorama base...\n");
	Mat mainPano = imgArray[0];

	//loop while stitching all images
	printf("loop while stitching all images...\n");
	for (int i = 1; i < num; i++)
	{
		//convert image to gray : computing faster
		printf("convert image to gray : computing faster...\n");
		Mat gray_mainImg, gray_objImg;
		cvtColor(mainPano, gray_mainImg, COLOR_RGB2GRAY);
		cvtColor(imgArray[i], gray_objImg, COLOR_RGB2GRAY);

		//detecting keypoints
		printf("detecting keypoints...\n");
		SiftFeatureDetector detector(0, 3, 0.3);
		vector<KeyPoint> point1, point2;
		detector.detect(gray_mainImg, point1);
		detector.detect(gray_objImg, point2);

		//computing descriptors
		printf("computing descriptors...\n");
		SiftDescriptorExtractor extractor;
		Mat descriptor1, descriptor2;
		extractor.compute(gray_mainImg, point1, descriptor1);
		extractor.compute(gray_objImg, point2, descriptor2);

		//match keypoints
		printf("match keypoints...\n");
		FlannBasedMatcher matcher;
		vector<DMatch> matches;
		matcher.match(descriptor1, descriptor2, matches);

		//get minimal distance
		printf("get minimal distance...\n");
		double mindistance = matches[0].distance;
		double distance;
		for (int i = 0; i < descriptor1.rows; i++) {
			distance = matches[i].distance;
			if (mindistance > distance)mindistance = distance;
		}

		//filtering matches using minimal distance
		printf("filtering matches using minimal distance...\n");
		vector<DMatch>goodmatch;
		for (int i = 0; i < descriptor1.rows; i++) {
			if (matches[i].distance < 5 * mindistance)
				goodmatch.push_back(matches[i]);
		}

		//drawing lines between matched keypoints
		printf("drawing lines between matched keypoints...\n");
		Mat matGoodMatcges;
		drawMatches(mainPano, point1, imgArray[i], point2, goodmatch, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//(option)shows the image
		printf("(option)shows the image...\n");
		imshow("after drawing matches " + std::to_string(i), matGoodMatcges);


		vector<Point2f> obj;
		vector<Point2f> scene;
		for (unsigned int i = 0; i < goodmatch.size(); i++) {
			obj.push_back(point1[goodmatch[i].queryIdx].pt);
			scene.push_back(point2[goodmatch[i].trainIdx].pt);
		}

		//Find Homography
		printf("Find Homography...\n");
		Mat homomatrix = findHomography(scene, obj, CV_RANSAC);

		//////////////////////////////////////////////////////
		////////////////////stitching part////////////////////
		//////////////////////////////////////////////////////

		//warp images
		printf("warp images...\n");
		Mat warp;
		warpPerspective(imgArray[i], warp, homomatrix, Size(imgArray[i].cols + mainPano.cols, imgArray[i].rows), INTER_CUBIC);

		//copy warped image
		printf("copy warped image...\n");
		Mat matPanorama;
		matPanorama = warp.clone();

		//paste
		printf("paste...\n");
		Mat matROI(matPanorama, Rect(0, 0, mainPano.cols, mainPano.rows));
		mainPano.copyTo(matROI);
		

		//eliminating black pixels by tracing column
		printf("eliminating black pixels by tracing column...\n");
		int max = 0;
		for (int i = 0; i < matPanorama.cols; ++i)
		{
			//   if not black: add to the list
			if (matPanorama.at<Vec3b>(matPanorama.rows * (3 / 4), i) != Vec3b(0, 0, 0))
			{
				if (max < i)max = i;
			}
			mainPano = matPanorama(Range(0, matPanorama.rows), Range(0, max));
		}
		imshow("after stitching " + std::to_string(i), mainPano);
	}
	return mainPano;
}
