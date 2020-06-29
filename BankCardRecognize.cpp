#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

void showImage(string name, Mat img);
Mat preProcess(Mat image, string image_name);
Mat preTemplateProcess(Mat image, string image_name);
vector<vector<Point>> find_contours(Mat image);
vector<Rect> sortContours(vector<vector<Point>> cont); //该函数返回的是个排好序的轮廓框
vector<Rect>  sortBoundRect(vector<Rect> boundRect); //这个是直接对轮廓框容器排序
vector<Rect> bubbleSort(vector<Rect> rect, int count);
vector<Rect> sortBoundRect(vector<Rect> boundRect);
int findIndex(vector<double> vec);
int main() {
	Mat tmp = imread("C:/Users/sherlock/Documents/template-matching-ocr/images/ocr_a_reference.png");
	showImage("tmp", tmp);
	Mat pre_tmp = preTemplateProcess(tmp, "template");
	vector<vector<Point>> contours = find_contours(pre_tmp);
	drawContours(tmp, contours, -1, Scalar(0, 0, 255), 3, 8);
	waitKey(60);
	cout << "have been finished find template contours" << endl;
	vector<Rect> boundRect = sortContours(contours);  //获得轮廓框容器
	//获得每个数字模板图片
	vector<Mat> aloneTmpImage(10);  //一定要初始化vector对象的大小
	for (int i = 0; i < boundRect.size(); i++) {		
		aloneTmpImage[i] = pre_tmp(boundRect[i]);
		resize(aloneTmpImage[i], aloneTmpImage[i], Size(60, 90));
		showImage("aloneTmpImage", aloneTmpImage[i]);
	}
	

	Mat bankCard = imread("C:/Users/sherlock/Documents/template-matching-ocr/images/credit_card_01.png");
	showImage("bankCard", bankCard);
	Mat bin_bankCard = preProcess(bankCard, "binaryBankCard_image");
	Mat rectKernel = getStructuringElement(MORPH_RECT, Size(9, 3)); //存疑为什么是这个size
	Mat sqKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat tophat_bankCard;
	medianBlur(bin_bankCard, bin_bankCard, 7);
	showImage("medianBlur", bin_bankCard);

	morphologyEx(bin_bankCard, tophat_bankCard, MORPH_TOPHAT, rectKernel, Point(-1, -1));
	dilate(tophat_bankCard, tophat_bankCard, sqKernel);
	dilate(tophat_bankCard, tophat_bankCard, sqKernel);
	showImage("tophat", tophat_bankCard);
	
	Mat dst;
	morphologyEx(tophat_bankCard, dst, MORPH_CLOSE, sqKernel);
	dilate(dst, dst, sqKernel);
	dilate(dst, dst, sqKernel);
	showImage("close1", dst);
	
	morphologyEx(dst, dst, MORPH_CLOSE, sqKernel);
	showImage("close2", dst);

	vector < vector<Point>> dst_contours = find_contours(dst);
	//此时的dst_contours包含了全部轮廓
	vector<Rect> testRect(20);
	for (int i = 0; i < dst_contours.size(); i++) {
		testRect[i] = boundingRect(dst_contours[i]);
	}
//得到信息: x:[40,173,302,430],y:[190,190,190,190],width:[110,107,109,112],height:[42,42,42,42]
	vector<Rect> dst_boundRect(20);  //存储轮廓框容器,暂时初始化为20
    vector<Rect> object_boundRect;  //存储目标轮廓框容器
	//auto object_boundRect;
	//寻找目标轮廓
	for (int i = 0; i < dst_contours.size(); i++) {
		dst_boundRect[i] = boundingRect(dst_contours[i]);
		float w = float(dst_boundRect[i].width);
		float h = float (dst_boundRect[i].height);
		double temp = w / h;
		if (temp > 2.5 & temp < 2.7) {
			if ((w > 100 & w < 120) & (h > 40 & h<45)) {
				object_boundRect.push_back(dst_boundRect[i]);
			}
		}
	}
	object_boundRect = sortBoundRect(object_boundRect);
	//遍历数字组合轮廓,共四个
	for (int i = 0; i < object_boundRect.size(); i++) {
		Mat groupNumber = bankCard(object_boundRect[i]);  //裁剪第i个
		groupNumber = preProcess(groupNumber, "groupNumbers");
		imshow("groupnumber", groupNumber);
		vector<vector<Point>> groupContours = find_contours(groupNumber);
		vector<Rect> groupRect = sortContours(groupContours); //这里应该是4个单数字轮廓
		for (int j = 0; j < groupRect.size(); j++) {   //遍历一个组合中的每个数字
			Mat number = groupNumber(groupRect[j]); //这里的groupNumber已经被预处理过了
			resize(number, number, Size(60, 90));
			showImage("number", number);
			vector<double> scores(10);  //用来存储0-9个模板的匹配结果,再求max可得最终结果
			for (int m = 0; m < aloneTmpImage.size(); m++) {  //遍历0-9模板,进行匹配
				Mat result;
				double maxVal;
				matchTemplate(number, aloneTmpImage[m], result, TM_CCOEFF_NORMED);
				minMaxLoc(result, NULL, &maxVal, NULL, NULL);
				scores[m] = maxVal;  //存储单个模板的匹配值
				cout <<"scores["<<m<<"]="<< scores[m] << endl;

			}
			double maxValue = *max_element(scores.begin(), scores.end());
			int maxPosition = findIndex(scores);
			cout << "maxValue= " << maxValue << "at [" << maxPosition << "]" << endl;
			rectangle(bankCard, object_boundRect[i].tl(), object_boundRect[i].br(), Scalar(0, 0, 255), 1);
			double numPosition_blx =object_boundRect[i].x + j*(object_boundRect[i].width/4.0) ;
			double numPosition_bly = object_boundRect[i].y;
			Point numPosition = Point (numPosition_blx, numPosition_bly);

			putText(bankCard, to_string(maxPosition), numPosition, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
			cout << numPosition << endl;
			showImage("lastResult", bankCard);

		}
	}
	showImage("LastResult", bankCard);
}

void showImage(string name, Mat img) {
	cv::imshow(name, img);
	cv::waitKey(1000);
}
Mat preTemplateProcess(Mat image, string image_name) {
	Mat dst;
	cvtColor(image, dst, COLOR_BGR2GRAY);
	imshow("cvtColor", dst);
	waitKey(100);
	threshold(dst, dst,110,255,THRESH_BINARY_INV);
	imshow("threshold", dst);
	waitKey(100);
	//showImage(image_name, dst);
	return dst;
}
Mat preProcess(Mat image, string image_name) {
	Mat dst;
	cvtColor(image, dst, COLOR_BGR2GRAY);
	imshow("cvtColor", dst);
	waitKey(100);
	threshold(dst, dst, 110, 255, THRESH_BINARY);
	imshow("threshold", dst);
	waitKey(100);
	//showImage(image_name, dst);
	return dst;
}
vector<vector<Point>> find_contours(Mat image) {
	vector<vector<Point>> contours;
	findContours(image, contours,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(image, contours, -1, Scalar(0, 0, 255),3,8);
	//showImage("image_contours", image);
	return contours;
}

vector<Rect> sortContours(vector<vector<Point>> contours) {
	int count = contours.size();
	vector<Rect> boundRect(count);  // 存储全部轮廓的容器boundRect
	for (int i = -0; i < count; i++) {
		boundRect[i] = boundingRect(Mat(contours[i]));
	}
	//从左往右排序
	boundRect = bubbleSort(boundRect, count);
	//遍历每一个轮廓
	return boundRect;

}
vector<Rect> sortBoundRect(vector<Rect> boundRect) {
	vector<Rect> sort_boundRect = boundRect;
	int count = sort_boundRect.size();  // 存储全部轮廓的容器boundRect
	sort_boundRect = bubbleSort(sort_boundRect, count);
	return sort_boundRect;
}

vector<Rect>  bubbleSort(vector<Rect> rect,  int count) {
	vector<Rect> sortRect = rect;
		for (int i = 0; i < count -1 ; i++) {
			for (int j = 0; j < count -1 -i; j++) {
				if (sortRect[j].x > sortRect[j+1].x) {        // 相邻元素两两对比
					Rect tmp = sortRect[j + 1];       // 元素交换
					sortRect[j + 1] = sortRect[j];
					sortRect[j] = tmp;
				}
			}
		}
	return sortRect;
}
int findIndex(vector<double> vec) {
	vector<double> temp = vec;
	double maxValue = temp[0];
	int index=0;
	for (int i = 0; i < temp.size(); i++) {
		if (temp[i] > maxValue) {
			maxValue = temp[i];
			index = i;
		}
	}
	return index;
}