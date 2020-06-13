#include<iostream>
#include<opencv.hpp>
#include <cstdio>

using namespace std;
using namespace cv;
const int num_frame = 2000;

#pragma warning(disable:4996)

//用kmeans对目标区域进行深度分割
Mat km_depth(const Mat& deproi,Mat result_d, double meand_bg,double meand_fg) {
	Mat SrcImage = deproi;
	int width = SrcImage.cols;
	int height = SrcImage.rows;
	int dims = SrcImage.channels();
	int sampleCount = width*height;
	int clusterCount = 2;
	Mat points(sampleCount, dims, CV_32F, Scalar(10));
	Mat labels;
	Mat centers(clusterCount, 1, points.type());
	int index = 0;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			Vec3b rgb = SrcImage.at<Vec3b>(row, col);
			points.at<float>(index, 0) = static_cast<int>(rgb[0]);
			points.at<float>(index, 1) = static_cast<int>(rgb[1]);
			points.at<float>(index, 2) = static_cast<int>(rgb[2]);
		}
	}
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0);
	kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);
	Mat result = Mat::zeros(SrcImage.size(), CV_8UC3);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			if (label == 1) {
				result.at<Vec3b>(row, col)[0] = 0;
				result.at<Vec3b>(row, col)[1] = 0;
				result.at<Vec3b>(row, col)[2] = 0;
			}
			else if (label == 0) {
				result.at<Vec3b>(row, col)[0] = 255;
				result.at<Vec3b>(row, col)[1] = 255;
				result.at<Vec3b>(row, col)[2] = 255;
			}
		}
	}

	double fg = 0, bg = 0, bgnum = 0, fgnum = 0;
	Mat result_g;
	cvtColor(SrcImage, result_g, COLOR_BGR2GRAY);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			if (label == 0) {
				bg = bg + result_g.at<uchar>(row, col);
			}
			else if (label == 1) {
				fg = fg + result_g.at<uchar>(row, col);
			}
		}
	}

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			if (label == 0) {
				result_d.at<uchar>(row, col) = bg;
				bgnum++;
			}
			else if (label == 1) {
				result_d.at<uchar>(row, col) = fg;
				fgnum++;
			}
		}
	}
	meand_bg = bg / bgnum;
	meand_fg = fg / fgnum;

	return result;
}

//初始化深度滤波器
Mat Dfilter_ini(const Mat& dframe, Rect& droi) {
	Mat depf;
	depf = dframe(droi);
	Mat resultd = Mat::zeros(depf.size(), CV_8UC1);
	double fg_d, bg_d;
	Mat D_km = km_depth(depf,resultd,bg_d,fg_d);
	double fg_p, bg_p;
	minMaxLoc(resultd, &bg_p, &fg_p);
	//imshow("DtoGray", D_km);	
}

Mat readindepth(int framenums) {
	
	char name[30];
	Mat Src = imread("1.jpg");
	Rect selectroi = selectROI("graywindow", Src, true, false);


	for (int i = 0; i < framenums; i++) {

		sprintf(name, "d/%d.jpg", i);
		Mat Src = imread(name);
		Mat Src = imread("1.jpg");
		Rect selectroi = selectROI("graywindow", Src, true, false);
		return Dfilter_ini(Src, selectroi);

	}


}

/*
bool readindepth(int i,Mat& depframe) {
	    char* name;
		sprintf(name, "dforc2//%d.jpg", i);
		depframe = imread(name);
		if (depframe.empty()) {
			return true;
		}
}

//用kmeans对目标区域进行深度分割
Mat km_depth(const Mat& deproi, double maxd, double mind, Point maxdp, Point mindp) {
	int width = deproi.cols;
	int height = deproi.rows;
	int dims = deproi.channels;
	if (dims != 1) {
		cvtColor(deproi, deproi, CV_BGR2GRAY);
		dims = 1;
	}
	int sampleCount = width*height;
	int clusterCount = 2;
	Mat points(sampleCount, dims, CV_32F, Scalar(10));
	Mat labels;
	Mat centers(clusterCount, 1, points.type());
	int index = 0;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int gray = deproi.at<int>(row, col);
			points.at<float>(index, 0) = static_cast<int>(gray);
		}
	}
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0);
	kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);
	Mat result = Mat::zeros(deproi.size(), CV_8U);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			if (label == 1) {
				result.at<Vec3b>(row, col)[0] = 0;
			}
			else if (label == 0) {
				result.at<Vec3b>(row, col)[0] = 255;
			}
		}
	}
	return result;

}

//初始化深度滤波器
bool Dfilter_ini(const Mat& dframe, Rect& droi) {
	int i = 0;
	Mat depf;
	
	depf = dframe(droi);
	double maxd, mind;
	Point maxdp, mindp;
    minMaxLoc(depf,&mind,&maxd,&mindp,&maxdp);
	depf.convertTo(depf,CV_32F);
	Mat D_km = km_depth(depf, mind,maxd, mindp,maxdp);



	for (int i = 0; i < num_frame; i++) {
		if(!readindepth(num_frame,depf))
			break;
	}
}
*/