#include<iostream>
#include<opencv.hpp>
#include <cstdio>

using namespace std;
using namespace cv; 

#pragma once

//depmask����ؼ���
Mat DepMask(Rect roi, Mat depthimage,float depthres);

//����depmask
float getDepth(Rect roi,Mat depthimage,Mat & depmask);

