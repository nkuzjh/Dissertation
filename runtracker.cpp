#include<iostream>
#include<opencv.hpp>
#include <cstdio>

#include"depth.h"

using namespace std;
using namespace cv;

//const int num_frame = 2000;
const int fps = 25;
const double eps = 0.00001;  //一个极小值，用于更新滤波器
const double learnrate = 0.2; // 滤波器模板学习率
const double thres = 7;  //PSR阈值
Mat haning;
Point2i ncenter, center;
Size2i nsize;
Rect newselectarea;
Mat G, H, A, B; //卷积公式H=A/B,G=F@H
double PSR;  //PSR 
Mat selectedf;
bool lostobject=false;

//深度mask
float depthres = 4;
float pre_depth;
float cur_depth;
float t0_depth;
bool occlusion=false;
Mat dep_mask;
float dist=0;

#pragma warning(disable:4996)

/*Point origin;
Rect selectarea;
bool selectif = false;

void onmouse(int event, int x, int y, int, void*) {
if (event == CV_EVENT_LBUTTONDOWN) {
selectarea.x = x;
selectarea.y = y;
selectif = false;
}
else if (event == CV_EVENT_LBUTTONUP) {
selectarea.width = x - selectarea.x;
selectarea.height = y - selectarea.y;
selectif = true;
}
}*/

//鼠标框选指定区域函数,返回选定区域的图像（矩阵）
//Mat mousearea() {}

//复数矩阵频谱显示
/*Mat VisualDft(cv::Mat input) {
Mat padded;                 //以0填充输入图像矩阵
int m = getOptimalDFTSize(input.rows); //getOptimalDFTSize函数返回给定向量尺寸的傅里叶最优尺寸大小。
int n = getOptimalDFTSize(input.cols);

//填充输入图像I，输入矩阵为padded，上方和左方不做填充处理
copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));//四个方向的常量0扩充

Mat re = Mat_<float>(padded), im = Mat::zeros(padded.size(), CV_32F);
Mat planes[] = { re, im };//两个矩阵
Mat complexI;
cv::merge(planes,2, complexI);     //将planes融合合并成一个多通道数组complexI

dft(complexI, complexI);        //进行傅里叶变换

//计算幅值，转换到对数尺度(logarithmic scale)
//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
split(complexI, planes);        //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
//即planes[0]为实部,planes[1]为虚部
magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude,求幅度谱
Mat magI = planes[0];

magI += Scalar::all(1);
log(magI, magI);                //转换到对数尺度(logarithmic scale)

//如果有奇数行或列，则对频谱进行裁剪
magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));  //magI.rows&-2得到不大于magI.rows的最大偶数

//重新排列傅里叶图像中的象限，使得原点位于图像中心
int cx = magI.cols / 2;
int cy = magI.rows / 2;

Mat q0(magI, Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
Mat q1(magI, Rect(cx, 0, cx, cy));      //右上角图像
Mat q2(magI, Rect(0, cy, cx, cy));      //左下角图像
Mat q3(magI, Rect(cx, cy, cx, cy));     //右下角图像

//变换左上角和右下角象限
Mat tmp;
q0.copyTo(tmp);
q3.copyTo(q0);
tmp.copyTo(q3);

//变换右上角和左下角象限
q1.copyTo(tmp);
q2.copyTo(q1);
tmp.copyTo(q2);

//归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
normalize(magI, magI, 0, 1, CV_MINMAX);

return magI;
}//图像的傅利叶变换。
/*

//连续图片序列转换为视频
/*void picTovideo(int fnum) {
Mat frame;
VideoWriter writer("E://ML//mosse//mosserealizing//Video.avi", CV_FOURCC('X', 'V', 'I', 'D'), 40, Size(640, 480));
int i = 0;
char picnames[1030];
string picname;
while (i < fnum) {
sprintf_s(picnames, "E://ML//Car1//Car1//img//%04d.jpg", i++);
picname = picnames;
frame = imread(picname);
if (!frame.data) {
cout << "could not load image file...\n" << endl;
}
if (!frame.empty()) {
imshow("picTovideo", frame);
}
writer << frame;
cout << i << endl;
waitKey(40);
}
}*/

//预处理目标区域图像
void preprocess(Mat& roi) {
	roi.convertTo(roi, CV_32F);//转换为32位float便于预处理
	log(roi + 1.0f, roi);//取对数，降低对比度
	Scalar mean, stddev;//scalar数据类型为向量，即一维矩阵（也可以理解为一维数组）
	meanStdDev(roi, mean, stddev);//meanstddev返回的值储存在矩阵中，使用mean和stddev是为了节省空间（向量是特殊的矩阵）！
	roi = (roi - mean[0]) / (stddev[0] + eps);//矩阵中各元素归一化（均值为0范数为1）eps为一个极小的值保证分母不为0
	createHanningWindow(haning, roi.size(), CV_32F);
	roi = roi.mul(haning);//创建hanning窗，处理目标区域,将图像使用余弦窗口处理，使边缘值逐渐归零

}

//小扰动下的随机仿射变换；直接用随机数生成器和放射变换会导致堆栈溢出
Mat affinewarp(const Mat& input) {
	RNG rng;
	double a = 0.1;
	double b = rng.uniform(-a, a);
	double c = cos(b), d = sin(b);

	Mat_<float> warpm(2, 3);
	warpm << rng.uniform(-a, a) + c, rng.uniform(-a, a) - d, 0,
		rng.uniform(-a, a) + d, rng.uniform(-a, a) + c, 0;//小范围的形变，无平移

	Mat_<float> warpm1(2, 1);
	warpm1 << input.cols / 2, input.rows / 2;
	warpm.col(2) = warpm1 - (warpm.colRange(0, 2))*warpm1;//平移变换

	Mat wrapini;
	warpAffine(input, wrapini, warpm, input.size(), BORDER_REFLECT);
	return wrapini;

}

//频域中复数矩阵的除法
Mat divindft(const Mat& A, const Mat& B) {  //return A/B; 等价于A*B^-1

	Mat Ari[2], Bri[2], deno, re, im;
	split(A, Ari);
	split(B, Bri);
	deno = Bri[0].mul(Bri[0]) + Bri[1].mul(Bri[1]);

	divide(Ari[0].mul(Bri[0]) + Ari[1].mul(Bri[1]), deno, re, 1.0);
	divide(Ari[1].mul(Bri[0]) + Ari[0].mul(Bri[1]), deno, im, -1.0);
	//按理说这个公式是divide（Ari[1].mul(Bri[0]) - Ari[0].mul(Bri[1]), deno, im, 1.0）才对

	Mat reandim[] = { re,im }, Hreim;
	merge(reandim, 2, Hreim);
	return Hreim;

}

//初始化滤波器
//初始化滤波器此函数是初始化A，B，H的值，先是将图像转化为灰度图，
//然后以目标框（boundingbox)的中心作傅里叶变换，目标框的初始傅里叶变换值G是将中间值设为1然后作高斯滤波得到，
//之后就多次代入如下公式对A，B初始化，初始化过程中学习速率η为1
void ini_filter(const Mat& firstf, Rect& firstroi,Mat depthframe) {
	
	//获取FFT的最优尺寸，根据最优尺寸，自动重新对第一帧图像选取兴趣区域，方便预处理后进行FFT
	//noting：矩阵在频域中做乘法需要扩充边界，
	//        在matlab/C++openCV中，FFT（频域）矩阵乘法自动采取循环的方式扩充边界后再计算两个矩阵的相乘(即在时域的卷积)
	//        因此无需单独编写扩展函数处理ROI（这里我最开始看论文的时候理解错了）
	//        且作者在2009CVPR的论文中提出了消除该运算引入的误差的方法
	int w = getOptimalDFTSize(int(firstroi.width));
	int h = getOptimalDFTSize(int(firstroi.height));
	//cout << "w=" << w << "h" << h << endl;
	int newx = int(floor((2 * firstroi.x - w + firstroi.width) / 2));
	int newy = int(floor((2 * firstroi.y - h + firstroi.height) / 2));
	ncenter.x = newx + w / 2;
	ncenter.y = newy + h / 2;
	nsize.width = w;
	nsize.height = h;

	Rect r(newx, newy, w, h);
	newselectarea = r;
	center = ncenter;//可能追踪一直失败的原因就在这里，根据response中最大值的位置选定新的区域位置，区域位置可能没有正确的传递下去
	dist = getDepth(newselectarea, depthframe,dep_mask);
	//cout  << " dist = " << dist << endl;
	cur_depth = dist;
	t0_depth = cur_depth;
	//cout << "cur_depth = " << cur_depth << endl;
	//cout << "pre_depth = " << pre_depth << endl;
	//cout << dep_mask.size() << endl;
	//cout << newselectarea.size() << endl;

	Mat dftroi;//新的适合做FFT的ROI区域
	getRectSubPix(firstf, nsize, ncenter, dftroi);

	Mat g = Mat::zeros(nsize, CV_32FC1);
	g.at<float>(h / 2, w / 2) = 255;//中心为1，其他值为0的矩阵

	Point gsforshow;
	cv::minMaxLoc(g, 0, 0, 0, &gsforshow);
	//cout << "nsize=" << nsize << endl;
	//cout << "gsforshow_initial=" << gsforshow << endl;
	cv::GaussianBlur(g, g, Size(-1, -1), 2.0);//高斯滤波

	double maxg;
	Point maxgforshow;
	cv::minMaxLoc(g, 0, &maxg, 0, &maxgforshow);//也可以使用minMaxIdx来返回最大/小值及其位置，minmaxidx主要用于多维矩阵
												//注意返回值的顺序，不能少，minmaxloc的顺序分别为输入矩阵，最小值，最大值，最小值位置，最大值位置,etc.
	g = g / maxg;//归一化初始的理想高斯矩阵

	cv::dft(g, G, DFT_COMPLEX_OUTPUT);//傅里叶变换

	//Mat gforshow;
	//idft(G, gforshow, DFT_SCALE | DFT_REAL_OUTPUT);
	//imshow("Gini",gforshow);

	A = Mat::zeros(G.size(), G.type());
	B = Mat::zeros(G.size(), G.type());
	for (int i = 0; i < 8; i++) {       //对第一帧图像进行仿射变换得到8个训练模板

		Mat iniaffineroi = affinewarp(dftroi);
		preprocess(iniaffineroi);

		Mat A_i, B_i, F_i;
		cv::dft(iniaffineroi, F_i, DFT_COMPLEX_OUTPUT);

		mulSpectrums(G, F_i, A_i, 0, true);
		mulSpectrums(F_i, F_i, B_i, 0, true);
		A += A_i;
		B += B_i;
	}
	H = divindft(A, B);//自定义的复数矩阵逐元素除法

}

void corelation(const Mat& Fc, Point& deltaxy) {

	Mat gc, Gc;

	//加入depmask
	//cout << dep_mask.size() << endl;
	//cout << H.size() << endl;
	//if (!occlusion) {
		//double D =sqrt( dep_mask.size().width * dep_mask.size().height);
		//H = D*dep_mask.mul(H);
	//}

	mulSpectrums(Fc, H, Gc, 0, true);
	cv::idft(Gc, gc, DFT_SCALE | DFT_REAL_OUTPUT);

	double gmax;
	Point newcenter;
	minMaxLoc(gc, 0, &gmax, 0, &newcenter);
	deltaxy.x = newcenter.x - int(gc.size().width / 2);
	deltaxy.y = newcenter.y - int(gc.size().height / 2);

	Scalar mean, std;
	meanStdDev(gc, mean, std);
	PSR = (gmax - mean[0]) / (std[0] + eps); // PSR 

}

//滤波器更新
void update_filter(const Mat& frame, Rect& targetarea,Mat depthframe) {
	
	pre_depth = cur_depth;
	dist = getDepth(targetarea, depthframe,dep_mask);
	imshow("dep_mask", dep_mask);
	cur_depth = dist;
	float d_value = cur_depth - pre_depth;
	if (d_value < 0)
		d_value = -d_value;

	//cout << "t0_depth = " << t0_depth << endl;
	//cout << "cur_depth = " << cur_depth << endl;
	//cout << "pre_depth = " << pre_depth << endl;
	//cout << "d_value : " << d_value << endl;
	//cout << "occlusion ?= " << occlusion << endl;

	Mat hu, fu;
	Mat Fu;

	getRectSubPix(frame, newselectarea.size(), center, selectedf, -1);

	//加入depmask
	//cout << dep_mask.size() << endl;
	//cout << selectedf.size() << endl;
	if (!occlusion) {
		dep_mask = dep_mask.mul(1 / 255);
		fu = selectedf.mul(dep_mask);
	}
	else
	    fu = selectedf;

	preprocess(fu);//预处理后续帧

	fu.convertTo(fu, CV_32F);//通道数不一样可以转换，但是矩阵元素的数据类型必须一样，默认的fu为CV_8U，这里要转换为32位单精度浮点数
	dft(fu, Fu, DFT_COMPLEX_OUTPUT);

	Point deltaxy;//corelation中找到的是中心点变化的值，并非中心点在整个图像矩阵的坐标
	corelation(Fu, deltaxy);//使用deltaxy传递这个坐标变化

	if ((!lostobject) && (PSR < thres)) {
		lostobject = true;
		cout << " -------------------- PSR is small ! ----------------------------" << endl;
		cout << "PSR = " << PSR << endl;
	}

	if ((!occlusion) && (d_value > depthres)){
		occlusion = true;
		t0_depth = pre_depth;
		cout << " **********************   occlusion  ************************* " << endl;
	}

	if (lostobject) {
		if (PSR > thres) {
			lostobject = false;
			cout << " ------------------------ PSR is normal ! -----------------------" << endl;
			cout << "PSR = " << PSR << endl;
		}
	}

	if (occlusion)
	{
		float dd_value = cur_depth - t0_depth;
		if (dd_value < 0)
			dd_value = -dd_value;
		if (dd_value < depthres)
		{
			occlusion = false;
			cout << " ********************  NO ******** occlusion  ****************** " << endl;
		}
	}

	if ((!occlusion) && (!lostobject)) {
		newselectarea.x += deltaxy.x;//使用deltaxy传递这个坐标变化，并更新newselectarea的左上角坐标
		newselectarea.y += deltaxy.y;
		center.x += deltaxy.x;
		center.y += deltaxy.y;

		Mat newroi;
		getRectSubPix(frame, newselectarea.size(), center, newroi);//根据更新后的坐标，重新框选物体区域

		if (newroi.channels() != 1)
			cvtColor(newroi, newroi, COLOR_BGR2GRAY);
		preprocess(newroi);

		Mat F, A_new, B_new;

		//mulSpectrums(Hu, Fu, G, 0, true);//G已经在corelation里面计算过了
		//G为上一帧的滤波器H与这一帧的输入图像F的响应

		dft(newroi, F, DFT_COMPLEX_OUTPUT);
		mulSpectrums(G, F, A_new, 0, true);//这里的G不是newroi处新一帧的F与上一帧的H的响应，是上一帧的F与上一帧的H的响应
		mulSpectrums(F, F, B_new, 0, true);

		A = (1 - learnrate)*A + learnrate*A_new;
		B = (1 - learnrate)*B + learnrate*B_new;
		H = divindft(A, B);
	}
	else if(occlusion && lostobject) 
		cout << " !!! CANNOT TRACK OBJECT !!! " << endl;
}


/*void pictovideo()
{
Mat testframe;
testframe = imread("ForMat//BlurCar4//img//0001.jpg");
Size size = testframe.size();
VideoWriter writer("ForMat//BlurCar4//video.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, size); //此处的尺寸针对不同的视频需要调整																		  //【2】循环显示每一帧
int i = 0;
char name[2000];
//  Mat frame;//定义一个Mat变量，用于存储每一帧的图像
while (i<num_frame)
{
i++;
sprintf(name, "ForMat//BlurCar4//img//%04d.jpg", i);
Mat frame = imread(name);
if (frame.empty())
{
break;
}
writer << frame;
}
}*/

/*void ToselectROI() {

cout << "Please input the initial ROI selectarea!" << endl;
cout << "input the left-top corner coordinates x = ?" << endl;
cin >> newselectarea.x;
cout << "input the left-top corner coordinates y = ?" << endl;
cin >> newselectarea.y;
cout << "input the width of ROI = ?" << endl;
cin >> newselectarea.width;
cout << "input the height of ROI = ?" << endl;
cin >> newselectarea.height;

}*/


/*bool readindepth(int i,Mat &depframe) {
char* name;
sprintf(name, "dforc2//%d.jpg", i);
depframe = imread(name);
if (depframe.empty()) {
return true;
}
}*/

void main() {

	//pictovideo();

	//输出跟踪区域坐标至文本中
	ofstream OutFile("Output.txt");
	//ofstream FailureOut("Basketball//FailureOutput.txt");

	//读取视频
	VideoCapture video("r_bear.avi");
	//car2,car4,carscale,fish,man,david,panda,woman
	Mat originframe;
	//Mat rotatef;
	Mat grayf;

	//显示第一帧图像
	video >> originframe;
	//Mat depf;
	//if(readindepth(1,depf))cv::imshow("dep",depf);
	cv::imshow("originwindow", originframe);

	//transpose(originframe, rotatef);
	//flip(rotatef, rotatef, 1);
	//rotatef = originframe;
	if (originframe.channels() != 1) {
		cvtColor(originframe, grayf, COLOR_BGR2GRAY);
	}
	else grayf = originframe;

	cv::imshow("graywindow", grayf);
	cv::moveWindow("graywindow", 1000, 0);

	//在第一帧图像中框选目标区域
	newselectarea = selectROI("graywindow", grayf, true, false);
	//为了与OTB2013的数据集做比较，由手动输入第一帧的ROI区域
	/*cout << "Please input the initial ROI selectarea!" << endl;
	cout << "input the left-top corner coordinates x = ?" << endl;
	cin >> newselectarea.x;
	cout << "input the left-top corner coordinates y = ?" << endl;
	cin >> newselectarea.y;
	cout << "input the width of ROI = ?" << endl;
	cin >> newselectarea.width;
	cout << "input the height of ROI = ?" << endl;
	cin >> newselectarea.height;*/
	//ToselectROI();

	//selectedf = grayf(newselectarea);

	//cv::imshow("ROI", selectedf);//显示第一帧中的ROI

	Mat depthframe1,depthframe;
	depthframe1 = imread("d_bear/1.jpg");
	cvtColor(depthframe1, depthframe, COLOR_BGR2GRAY);

	//cout << grayf.channels() << endl;
	//cout << depthframe.channels() << endl;
	
	//imshow("inidepframe", depthframe);
	//Size depthframeS = depthframe.size();
	char depthname[30];

	//根据框选区域和第一帧图像，初始化滤波器
	ini_filter(grayf, newselectarea,depthframe);

	//捕捉鼠标，根据鼠标移动轨迹，使用onmouse函数得到在后续帧中画出矩形框所需的参数
	//setMouseCallback("graywindow", onmouse, 0);

	//时钟函数获取运行时间
	//clock_t begin, end;
	//begin = clock();
	clock_t extrabegin, extraend;
	double extraruntime = 0;

	//读取视频，利用循环结构显示每一帧
	int i = 0;
	while (1) {

		i++;
		cout << i << endl;
		//cout << "PSR = " << PSR << endl;

		sprintf(depthname, "d_bear/%d.jpg", i);
		depthframe1 = imread(depthname);
		cvtColor(depthframe1, depthframe, COLOR_BGR2GRAY);
		//imshow("inidepframe", depthframe);

		//视频存入frame矩阵
		video >> originframe;

		//读取完成则退出
		if (originframe.empty() == true)
			break;

		//窗口显示读入视频的新的一帧
		cv::imshow("originwindow", originframe);

		//还可以利用IpIImage格式下的左上角坐标旋转图像IpIImage* originf=originframe; originf->origin = 0;
		//利用transpose()和flip()旋转图像
		//transpose()可图像矩阵转置，flip()可将矩阵按x轴/y轴/原点翻转
		//transpose(originframe, rotatef);
		//flip(rotatef, rotatef, 1);
		//rotatef = originframe;

		//利用cvtColor函数转换为灰度图
		//COLOR_BGR2GRAY为灰度图的色彩空间转换模式（code）
		if (originframe.channels() != 1) {
			cvtColor(originframe, grayf, COLOR_BGR2GRAY);
		}
		else grayf = originframe;

		extrabegin = clock();
		update_filter(grayf, newselectarea,depthframe);
		extraend = clock();

		//if (newselectarea.x > 0) 
		OutFile << newselectarea.x;
		//else OutFile << 0;
		OutFile.write(" ", 1);
		//if (newselectarea.y > 0)
		OutFile << newselectarea.y;
		//else OutFile << 0;
		OutFile.write(" ", 1);
		OutFile << newselectarea.width;
		OutFile.write(" ", 1);
		OutFile << newselectarea.height;
		OutFile.write("\n", 1);

		//if (occlusion && lostobject) {
		//	cout << " WARNING: Track Failure in The " <<i+1<<"th Frame ! "<< endl;
		//}

		/*if (occlusion) {
			cout << "Detect Occlusion in the " << i + 1 << "th frame" << endl;

			//OutFile << "tracking failure in the frame ";
			//OutFile << i+1;
			//OutFile.write("\n", 1);
			//OutFile << "PSR =";
			//OutFile << PSR;
			//OutFile.write("\n", 1);

		}
		else if ((occlusion) && PSR < thres) {
			cout << "PSR = " << PSR << endl;
			cout << "Lost Object in the " << i + 1 << "th frame" << endl;
		}*/

			//新建窗口仅显示框选区域
			Mat selectedfforshow;
			getRectSubPix(grayf, newselectarea.size(), center, selectedfforshow, -1);

			//在graywindow窗口显示带追踪框的灰度图
			rectangle(grayf, newselectarea, Scalar(255, 255, 0), 2, 8, 0);
			cv::imshow("graywindow", grayf);
			cv::moveWindow("graywindow", 1000, 0);

		
		//根据fps显示视频
		int key = cv::waitKey(25);

		extraruntime += double((1000 * (extraend - extrabegin) / CLOCKS_PER_SEC));
		//cout << "the frame" << " " << i << " " << "is finished!" << endl;
		//按下esc退出显示窗口
		//if (key == 27)
		//	break;
	}

	OutFile.close();

	//输出运行时间
	//end = clock();
	//double runtime = double(1000 * (end - begin) / CLOCKS_PER_SEC) - extraruntime;
	double runtime = extraruntime;
	/*OutFile << "runtime = ";
	OutFile<<runtime;
	OutFile << " ms";
	OutFile.write("\n", 1);
	OutFile << "runtime = ";
	OutFile << runtime/(1000*i);
	OutFile << " second/per frame";
	OutFile.write("\n", 1);
	OutFile << "runtime = ";
	OutFile << i*1000/runtime;
	OutFile << " frame/per second";*/

	cout << "TRACK END" << endl;

	cout << "runtime = " << runtime << " ms " << endl;
	cout << "runtime = " << runtime / (1000 * i) << " second/per frame" << endl;
	cout << "runtime = " << i * 1000 / runtime << " frame/per second" << endl;

	cv::waitKey();

}
