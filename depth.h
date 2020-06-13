#include<iostream>
#include<opencv.hpp>
#include <cstdio>

using namespace std;
using namespace cv; 

#pragma once

//depmask的相关计算
Mat DepMask(Rect roi, Mat depthimage,float depthres);

//返回depmask
float getDepth(Rect roi,Mat depthimage,Mat & depmask);

