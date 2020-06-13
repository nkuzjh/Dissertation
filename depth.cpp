#include<iostream>
#include<opencv.hpp>
#include <cstdio>

#include"depth.h"

using namespace std;
using namespace cv;

#pragma once


Mat DepMask(Rect roi, Mat depthimage, float depthres) {
	Mat deproi = depthimage(roi);
	Mat depmask = Mat::zeros(deproi.size(), CV_8U);
	threshold(deproi, depmask,depthres,255,THRESH_BINARY);
	return depmask;
};


float getDepth(Rect roi, Mat depthimage,Mat & depmask) {

	imshow("depthframe", depthimage);

	ofstream fout("depthresult.txt");

	Mat deproi = depthimage(roi);
	
	vector<vector<float>> result;

	for (int i = 0; i < 255; i++)
	{
		vector<float> v;
		result.push_back(v);
	}
	//result即为深度直方图
	//这里用（maxdist-mindist）/interval初始化result中bin的数量

	//长度为result中bin的数量的数组
	//int count[result.size()];
	int count[255];

	int row_begin; //= roi_y + roi.height / 3;
	int col_begin; //= roi_x + roi.width / 3;
	int row_end; //= roi_y + 2 * roi.height / 3;
	int col_end; //= roi_x + 2 * roi.width / 3;

	if (roi.y <= 0)
		row_begin = 0;
	else if (roi.y >= (depthimage.rows - 10))
		return 0;
	else
		row_begin = roi.y;

	if (roi.y + roi.height <= 0)
		return 0;
	else if (roi.y + roi.height >= depthimage.rows)
		row_end = depthimage.rows;
	else
		row_end = roi.y + roi.height;

	if (roi.x <= 0)
		col_begin = 0;
	else if (roi.x >= depthimage.cols)
		return 0;
	else
		col_begin = roi.x;

	if (roi.x + roi.width <= 0)
		return 0;
	else if (roi.x + roi.width >= depthimage.cols)
		col_end = depthimage.cols;
	else
		col_end = roi.x + roi.width;

	for (int row = row_begin; row < row_end; row++)
	{
		for (int col = col_begin; col < col_end; col++)
		{
			if (depthimage.at<uchar>(row, col) > 0)
			{
				float depth = depthimage.at<uchar>(row, col);
				int inter = int(depth);
				result.at(inter).push_back(depth);
			}
		}
	}

	//把深度图像各像素的深度放入相应bin中，形成深度直方图
	int maxSize = 0;
	int index = 0;
	if (fout.is_open())
	{
		for (int i = 0; i < result.size(); i++)
		{
			fout << "vector " << i << " size " << " " << result.at(i).size() << endl;
			if (result.at(i).size() > maxSize)
			{
				maxSize = result.at(i).size();
				index = i;
			}
			if (i < 2)
			{
				count[i] = result.at(i).size() + result.at(i + 1).size() + result.at(i + 2).size();
			}
			else if (i >= 2 && i < result.size() - 2)
			{
				count[i] = result.at(i - 2).size() + result.at(i - 1).size() + result.at(i).size() +
					result.at(i + 1).size() + result.at(i + 2).size();
			}
			else
			{
				count[i] = result.at(i - 2).size() + result.at(i - 1).size() + result.at(i).size();
			}
			fout << "count " << i << " " << count[i] << endl << endl;
		}
		fout << "index : " << index << "  maxSize : " << maxSize << endl;
	}
	//count数组中的数据为：
	//对应bin以及前后两个bin中像素点的个数和
	//result.at(i).size就是深度直方图中纵轴的大小，count[i]是i附近连续五个bin求和的大小

	int maxCount = 0;
	int indexCount = 0;
	for (int i = 0; i < result.size(); i++)
	{
		if (count[i] > maxCount)
		{
			maxCount = count[i];
			indexCount = i;
		}
		fout << "count " << i << " : " << count[i] << endl;
	}

	fout << "indexCount " << indexCount << " maxCount : " << maxCount << endl;

	//找count[i]中最大的值
	//###############################
	//###############################
	//indexcount记录了count[i]中最大值在数组中的位置，即对应的i
	//###############################
	//###############################

	float distance = 0;
	// for(int i = 0; i < result.at(indexCount).size(); i++)
	// {
	//     distance += result.at(indexCount).at(i);
	// }
	// distance /= result.at(indexCount).size();
	if (indexCount < 2)
		//上面我们可以知道前两个和后两个count[i]中只储存了3个bin的size()
	{
		for (int i = indexCount; i <= indexCount + 2; i++)
		{
			for (int j = 0; j < result.at(i).size(); j++)
				distance += result.at(i).at(j);
		}
		distance /= count[indexCount];
		//distance为实际深度，等于三个bin的总深度除以三个bin中的像素数量
	}
	else if (indexCount >= 2 && indexCount < result.size() - 2)
	{
		for (int i = indexCount - 2; i <= indexCount + 2; i++)
		{
			for (int j = 0; j < result.at(i).size(); j++)
				distance += result.at(i).at(j);
		}
		distance /= count[indexCount];
		//distance为实际深度，等于五个bin的总深度除以五个bin中的像素数量
	}
	else
	{
		for (int i = indexCount - 2; i <= indexCount; i++)
		{
			for (int j = 0; j < result.at(i).size(); j++)
				distance += result.at(i).at(j);
		}
		distance /= count[indexCount];
		//distance为实际深度，等于三个bin的总深度除以三个bin中的像素数量
	}

	fout.close();
	// cout << " distance " << distance << endl;

	float depthres=distance;
	if (indexCount < 2)
		//上面我们可以知道前两个和后两个count[i]中只储存了3个bin的size()
	{
		for (int i = indexCount; i <= indexCount + 2; i++)
		{
			for (int j = 0; j < result.at(i).size(); j++)
				if (depthres< result.at(i).at(j))
					depthres = result.at(i).at(j);
		}
	}
	else if (indexCount >= 2 && indexCount < result.size() - 2)
	{
		for (int i = indexCount - 2; i <= indexCount + 2; i++)
		{
			for (int j = 0; j < result.at(i).size(); j++)
				if (depthres< result.at(i).at(j))
					depthres = result.at(i).at(j);
		}
	}
	else
	{
		for (int i = indexCount - 2; i <= indexCount; i++)
		{
			for (int j = 0; j < result.at(i).size(); j++)
				if (depthres< result.at(i).at(j))
					depthres = result.at(i).at(j);
		}
	}

	depmask=DepMask(roi,depthimage,depthres);
	//threshold(deproi, depmask,depthres,255,THRESH_BINARY);

	//cout << "depthres = " << depthres << " distance = " << *distance << endl;
	//imshow("depmask", depmask);

	//认为最大值（认为result中的值越大物体距离相机越近）即为目标的深度值distance（公认：在第一帧中，目标是距离相机最近的物体）
	return distance;
}


