#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>


using namespace cv;
using namespace std;


void ConvertVectortoMatrix(vector<vector<float>> &testHOG, Mat &testMat);
vector<float> find_HOG_feature_image(Mat img);


/////////////////////////////////////////////
//*****please setting the number of images*****
#define NUMBER_train 450
#define NUMBER_test 50
/////////////////////////////////////////////


int main()
{
	vector<vector<float>> HOG_test_data;
	vector<int> HOG_test_data_label;

	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "test Image load" << endl;

	for (int num = 0; num < 10; num++)
	{
		cout << num << " image load" << endl;
		for (int i = NUMBER_train; i < NUMBER_test + NUMBER_train; i++)
		{
			Mat MNIST = imread("./data1/MNIST/data " + std::to_string(num) + "_" + std::to_string(i + 1) + ".PNG", COLOR_RGB2GRAY);
			if (!(MNIST.data))
			{
				cout << "image load fail" << endl;
				return 0;
			}
			HOG_test_data.push_back(find_HOG_feature_image(MNIST));
			HOG_test_data_label.push_back(num);
		}

	}

	cout << "\n//////////////////////////////////////////////////////" << endl;


	int descriptor_size = HOG_test_data[0].size();

	Mat HOG_test_data_Mat(HOG_test_data.size(), descriptor_size, CV_32FC1);
	Mat HOG_test_data_label_Mat(HOG_test_data_label.size(), 1, CV_32FC1);

	ConvertVectortoMatrix(HOG_test_data, HOG_test_data_Mat);

	for (int i = 0; i < HOG_test_data_label.size(); i++)
	{
		HOG_test_data_label_Mat.at<float>(i, 0) = HOG_test_data_label[i];
	}


	cout << "\n//////////////////////////////////////////////////////" << endl;
	cout << "                      svm load                      " << endl;
	cout << "//////////////////////////////////////////////////////\n" << endl;

	CvSVM svm;
	svm.load("./MNIST_HOG_SVM_Linear_acc97.XML");

	cout << "\n//////////////////////////////////////////////////////" << endl;
	cout << "                      svm predict                      " << endl;
	cout << "//////////////////////////////////////////////////////\n" << endl;

	Mat Response_test;

	svm.predict(HOG_test_data_Mat, Response_test);
	float count_test = 0, accuracy_test = 0;
	for (int i = 0; i < Response_test.rows; i++)
	{
		if (Response_test.at<float>(i, 0) == HOG_test_data_label[i])
		{
			count_test = count_test + 1;
		}
	}

	accuracy_test = (count_test / Response_test.rows) * 100;
	cout << "accuracy_test : " << accuracy_test << endl;

}

vector<float> find_HOG_feature_image(Mat img)
{
	HOGDescriptor IMAGE_HOG
	(
		Size(20, 20), //winSize
		Size(8, 8), //blocksize
		Size(4, 4), //blockStride,
		Size(8, 8), //cellSize,
		9, //nbins,
		1, //derivAper,
		-1, //winSigma,
		0, //histogramNormType,
		0.2, //L2HysThresh,
		0,//gammal correction,
		64//nlevels=64
	);



	vector<float> hog_descriptor;
	IMAGE_HOG.compute(img, hog_descriptor);
	return hog_descriptor;

}

void ConvertVectortoMatrix(vector<vector<float>> &testHOG, Mat &testMat)
{

	int descriptor_size = testHOG[0].size();

	for (int i = 0; i < testHOG.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			testMat.at<float>(i, j) = testHOG[i][j];
		}
	}
}
