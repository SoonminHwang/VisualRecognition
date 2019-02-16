#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void ConvertVectortoMatrix(vector<vector<float> > &In_vector, Mat &Out_Mat);
vector<vector<float>> CsvtoVector(String filename);

int main()
{
	vector<vector<float>> CNN_train_data_label = CsvtoVector("./image_train_label_rs.csv");
	vector<vector<float>> CNN_train_data = CsvtoVector("./image_train_rs.csv");
	vector<vector<float>> CNN_test_data_label = CsvtoVector("./image_test_label_rs.csv");
	vector<vector<float>> CNN_test_data = CsvtoVector("./image_test_rs.csv");

	int descriptor_size = CNN_train_data[0].size();

	Mat MNIST_CNN_train_data_Mat(CNN_train_data.size(), descriptor_size, CV_32FC1);
	Mat MNIST_CNN_train_data_label_Mat(CNN_train_data_label.size(), 1, CV_32FC1);
	Mat MNIST_CNN_test_data_Mat(CNN_test_data.size(), descriptor_size, CV_32FC1);
	Mat MNIST_CNN_test_data_label_Mat(CNN_test_data_label.size(), 1, CV_32FC1);

	ConvertVectortoMatrix(CNN_train_data, MNIST_CNN_train_data_Mat);
	ConvertVectortoMatrix(CNN_test_data, MNIST_CNN_test_data_Mat);

	for (int i = 0; i < CNN_train_data_label.size(); i++)
	{
		MNIST_CNN_train_data_label_Mat.at<float>(i, 0) = CNN_train_data_label[i][0];
	}
	for (int i = 0; i < CNN_test_data_label.size(); i++)
	{
		MNIST_CNN_test_data_label_Mat.at<float>(i, 0) = CNN_test_data_label[i][0];
	}


	cout << "\n//////////////////////////////////////////////////////" << endl;
	cout << "                      svm setting                      " << endl;
	cout << "//////////////////////////////////////////////////////\n" << endl;

	CvSVM svm;
	
	 CvSVMParams params = CvSVMParams
	 (
	  CvSVM::C_SVC,   // Type of SVM, here N classes (see manual)
	  CvSVM::LINEAR,  // kernel type (see manual)
	  0.0,            // kernel parameter (degree) for poly kernel only
	  0.0,            // kernel parameter (gamma) for poly/rbf kernel only
	  0.0,            // kernel parameter (coef0) for poly/sigmoid kernel only
	  10,             // SVM optimization parameter C
	  0,              // SVM optimization parameter nu (not used for N classe SVM)
	  0,              // SVM optimization parameter p (not used for N classe SVM)
	  NULL,           // class wieghts (or priors)
	  // Optional weights, assigned to particular classes.
	  // They are multiplied by C and thus affect the misclassification
	  // penalty for different classes. The larger weight, the larger penalty
	  // on misclassification of data from the corresponding class.

	  // termination criteria for learning algorithm

	  cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)
    );

	cout << "\n//////////////////////////////////////////////////////" << endl;
	cout << "                      svm train                      " << endl;
	cout << "//////////////////////////////////////////////////////\n" << endl;
	
	svm.train_auto(MNIST_CNN_train_data_Mat, MNIST_CNN_train_data_label_Mat, Mat(), Mat(), params, 10);

	cout << "\n//////////////////////////////////////////////////////" << endl;
	cout << "                      svm predict                      " << endl;
	cout << "//////////////////////////////////////////////////////\n" << endl;

	Mat Response_test;
	Mat Response_train;


	svm.predict(MNIST_CNN_test_data_Mat, Response_test);
	float count_test = 0, accuracy_test = 0;

	for (int i = 0; i < Response_test.rows; i++)
	{
		if (Response_test.at<float>(i, 0) == CNN_test_data_label[i][0])
		{
			count_test = count_test + 1;
		}
	}

	svm.predict(MNIST_CNN_test_data_Mat, Response_train);
	float count_train = 0, accuracy_train = 0;

	for (int i = 0; i < Response_train.rows; i++)
	{
		if (Response_train.at<float>(i, 0) == CNN_train_data_label[i][0])
		{
			count_train = count_train + 1;
		}
	}

	accuracy_test = (count_test / Response_test.rows) * 100;
	cout << "accuracy_test : " << accuracy_test << endl;
	accuracy_train = (count_train / Response_train.rows) * 100;
	cout << "accuracy_train : " << accuracy_train << endl;


	cout << "\n//////////////////////////////////////////////////////" << endl;
	cout << "                      svm save                   " << endl;
	cout << "//////////////////////////////////////////////////////\n" << endl;
	/// Save svm file
	svm.save("MNIST_CNN_SVM.xml");

}

vector<vector<float>> CsvtoVector(String filename)
{
	ifstream train_data(filename);
	vector<vector<float> > vector_data;

	if (!train_data.is_open())
	{
		cout << "Error: File Opne" << endl;
	}
	else
	{
		cout << "Find : " << filename << endl;
		cout << "starting loading" << endl;
	}

	if (train_data)
	{
		int cnt = 0;
		string line;
		while (getline(train_data, line))
		{
			vector<float> train_data_vector_2;
			stringstream linStream(line);
			string cell;
			cnt++;
			while (getline(linStream, cell, ','))
			{
				train_data_vector_2.push_back(stof(cell));
			}
			
			vector_data.push_back(train_data_vector_2);
		}
	}
	cout << "finished loading " << endl<<endl;

	return vector_data;
}

void ConvertVectortoMatrix(vector<vector<float> > &In_vector, Mat &Out_Mat)
{

	int descriptor_size = In_vector[0].size();

	for (int i = 0; i < In_vector.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			Out_Mat.at<float>(i, j) = In_vector[i][j];
		}
	}

}
