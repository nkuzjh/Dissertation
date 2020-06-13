#include<iostream>
#include<opencv.hpp>
#include <cstdio>

#include"depth.h"

using namespace std;
using namespace cv;

//const int num_frame = 2000;
const int fps = 25;
const double eps = 0.00001;  //һ����Сֵ�����ڸ����˲���
const double learnrate = 0.2; // �˲���ģ��ѧϰ��
const double thres = 7;  //PSR��ֵ
Mat haning;
Point2i ncenter, center;
Size2i nsize;
Rect newselectarea;
Mat G, H, A, B; //�����ʽH=A/B,G=F@H
double PSR;  //PSR 
Mat selectedf;
bool lostobject=false;

//���mask
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

//����ѡָ��������,����ѡ�������ͼ�񣨾���
//Mat mousearea() {}

//��������Ƶ����ʾ
/*Mat VisualDft(cv::Mat input) {
Mat padded;                 //��0�������ͼ�����
int m = getOptimalDFTSize(input.rows); //getOptimalDFTSize�������ظ��������ߴ�ĸ���Ҷ���ųߴ��С��
int n = getOptimalDFTSize(input.cols);

//�������ͼ��I���������Ϊpadded���Ϸ����󷽲�����䴦��
copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));//�ĸ�����ĳ���0����

Mat re = Mat_<float>(padded), im = Mat::zeros(padded.size(), CV_32F);
Mat planes[] = { re, im };//��������
Mat complexI;
cv::merge(planes,2, complexI);     //��planes�ںϺϲ���һ����ͨ������complexI

dft(complexI, complexI);        //���и���Ҷ�任

//�����ֵ��ת���������߶�(logarithmic scale)
//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
split(complexI, planes);        //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
//��planes[0]Ϊʵ��,planes[1]Ϊ�鲿
magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude,�������
Mat magI = planes[0];

magI += Scalar::all(1);
log(magI, magI);                //ת���������߶�(logarithmic scale)

//����������л��У����Ƶ�׽��вü�
magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));  //magI.rows&-2�õ�������magI.rows�����ż��

//�������и���Ҷͼ���е����ޣ�ʹ��ԭ��λ��ͼ������
int cx = magI.cols / 2;
int cy = magI.rows / 2;

Mat q0(magI, Rect(0, 0, cx, cy));       //���Ͻ�ͼ�񻮶�ROI����
Mat q1(magI, Rect(cx, 0, cx, cy));      //���Ͻ�ͼ��
Mat q2(magI, Rect(0, cy, cx, cy));      //���½�ͼ��
Mat q3(magI, Rect(cx, cy, cx, cy));     //���½�ͼ��

//�任���ϽǺ����½�����
Mat tmp;
q0.copyTo(tmp);
q3.copyTo(q0);
tmp.copyTo(q3);

//�任���ϽǺ����½�����
q1.copyTo(tmp);
q2.copyTo(q1);
tmp.copyTo(q2);

//��һ��������0-1֮��ĸ�����������任Ϊ���ӵ�ͼ���ʽ
normalize(magI, magI, 0, 1, CV_MINMAX);

return magI;
}//ͼ��ĸ���Ҷ�任��
/*

//����ͼƬ����ת��Ϊ��Ƶ
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

//Ԥ����Ŀ������ͼ��
void preprocess(Mat& roi) {
	roi.convertTo(roi, CV_32F);//ת��Ϊ32λfloat����Ԥ����
	log(roi + 1.0f, roi);//ȡ���������ͶԱȶ�
	Scalar mean, stddev;//scalar��������Ϊ��������һά����Ҳ�������Ϊһά���飩
	meanStdDev(roi, mean, stddev);//meanstddev���ص�ֵ�����ھ����У�ʹ��mean��stddev��Ϊ�˽�ʡ�ռ䣨����������ľ��󣩣�
	roi = (roi - mean[0]) / (stddev[0] + eps);//�����и�Ԫ�ع�һ������ֵΪ0����Ϊ1��epsΪһ����С��ֵ��֤��ĸ��Ϊ0
	createHanningWindow(haning, roi.size(), CV_32F);
	roi = roi.mul(haning);//����hanning��������Ŀ������,��ͼ��ʹ�����Ҵ��ڴ���ʹ��Եֵ�𽥹���

}

//С�Ŷ��µ��������任��ֱ����������������ͷ���任�ᵼ�¶�ջ���
Mat affinewarp(const Mat& input) {
	RNG rng;
	double a = 0.1;
	double b = rng.uniform(-a, a);
	double c = cos(b), d = sin(b);

	Mat_<float> warpm(2, 3);
	warpm << rng.uniform(-a, a) + c, rng.uniform(-a, a) - d, 0,
		rng.uniform(-a, a) + d, rng.uniform(-a, a) + c, 0;//С��Χ���α䣬��ƽ��

	Mat_<float> warpm1(2, 1);
	warpm1 << input.cols / 2, input.rows / 2;
	warpm.col(2) = warpm1 - (warpm.colRange(0, 2))*warpm1;//ƽ�Ʊ任

	Mat wrapini;
	warpAffine(input, wrapini, warpm, input.size(), BORDER_REFLECT);
	return wrapini;

}

//Ƶ���и�������ĳ���
Mat divindft(const Mat& A, const Mat& B) {  //return A/B; �ȼ���A*B^-1

	Mat Ari[2], Bri[2], deno, re, im;
	split(A, Ari);
	split(B, Bri);
	deno = Bri[0].mul(Bri[0]) + Bri[1].mul(Bri[1]);

	divide(Ari[0].mul(Bri[0]) + Ari[1].mul(Bri[1]), deno, re, 1.0);
	divide(Ari[1].mul(Bri[0]) + Ari[0].mul(Bri[1]), deno, im, -1.0);
	//����˵�����ʽ��divide��Ari[1].mul(Bri[0]) - Ari[0].mul(Bri[1]), deno, im, 1.0���Ŷ�

	Mat reandim[] = { re,im }, Hreim;
	merge(reandim, 2, Hreim);
	return Hreim;

}

//��ʼ���˲���
//��ʼ���˲����˺����ǳ�ʼ��A��B��H��ֵ�����ǽ�ͼ��ת��Ϊ�Ҷ�ͼ��
//Ȼ����Ŀ���boundingbox)������������Ҷ�任��Ŀ���ĳ�ʼ����Ҷ�任ֵG�ǽ��м�ֵ��Ϊ1Ȼ������˹�˲��õ���
//֮��Ͷ�δ������¹�ʽ��A��B��ʼ������ʼ��������ѧϰ���ʦ�Ϊ1
void ini_filter(const Mat& firstf, Rect& firstroi,Mat depthframe) {
	
	//��ȡFFT�����ųߴ磬�������ųߴ磬�Զ����¶Ե�һ֡ͼ��ѡȡ��Ȥ���򣬷���Ԥ��������FFT
	//noting��������Ƶ�������˷���Ҫ����߽磬
	//        ��matlab/C++openCV�У�FFT��Ƶ�򣩾���˷��Զ���ȡѭ���ķ�ʽ����߽���ټ���������������(����ʱ��ľ��)
	//        ������赥����д��չ��������ROI���������ʼ�����ĵ�ʱ�������ˣ�
	//        ��������2009CVPR�������������������������������ķ���
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
	center = ncenter;//����׷��һֱʧ�ܵ�ԭ������������response�����ֵ��λ��ѡ���µ�����λ�ã�����λ�ÿ���û����ȷ�Ĵ�����ȥ
	dist = getDepth(newselectarea, depthframe,dep_mask);
	//cout  << " dist = " << dist << endl;
	cur_depth = dist;
	t0_depth = cur_depth;
	//cout << "cur_depth = " << cur_depth << endl;
	//cout << "pre_depth = " << pre_depth << endl;
	//cout << dep_mask.size() << endl;
	//cout << newselectarea.size() << endl;

	Mat dftroi;//�µ��ʺ���FFT��ROI����
	getRectSubPix(firstf, nsize, ncenter, dftroi);

	Mat g = Mat::zeros(nsize, CV_32FC1);
	g.at<float>(h / 2, w / 2) = 255;//����Ϊ1������ֵΪ0�ľ���

	Point gsforshow;
	cv::minMaxLoc(g, 0, 0, 0, &gsforshow);
	//cout << "nsize=" << nsize << endl;
	//cout << "gsforshow_initial=" << gsforshow << endl;
	cv::GaussianBlur(g, g, Size(-1, -1), 2.0);//��˹�˲�

	double maxg;
	Point maxgforshow;
	cv::minMaxLoc(g, 0, &maxg, 0, &maxgforshow);//Ҳ����ʹ��minMaxIdx���������/Сֵ����λ�ã�minmaxidx��Ҫ���ڶ�ά����
												//ע�ⷵ��ֵ��˳�򣬲����٣�minmaxloc��˳��ֱ�Ϊ���������Сֵ�����ֵ����Сֵλ�ã����ֵλ��,etc.
	g = g / maxg;//��һ����ʼ�������˹����

	cv::dft(g, G, DFT_COMPLEX_OUTPUT);//����Ҷ�任

	//Mat gforshow;
	//idft(G, gforshow, DFT_SCALE | DFT_REAL_OUTPUT);
	//imshow("Gini",gforshow);

	A = Mat::zeros(G.size(), G.type());
	B = Mat::zeros(G.size(), G.type());
	for (int i = 0; i < 8; i++) {       //�Ե�һ֡ͼ����з���任�õ�8��ѵ��ģ��

		Mat iniaffineroi = affinewarp(dftroi);
		preprocess(iniaffineroi);

		Mat A_i, B_i, F_i;
		cv::dft(iniaffineroi, F_i, DFT_COMPLEX_OUTPUT);

		mulSpectrums(G, F_i, A_i, 0, true);
		mulSpectrums(F_i, F_i, B_i, 0, true);
		A += A_i;
		B += B_i;
	}
	H = divindft(A, B);//�Զ���ĸ���������Ԫ�س���

}

void corelation(const Mat& Fc, Point& deltaxy) {

	Mat gc, Gc;

	//����depmask
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

//�˲�������
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

	//����depmask
	//cout << dep_mask.size() << endl;
	//cout << selectedf.size() << endl;
	if (!occlusion) {
		dep_mask = dep_mask.mul(1 / 255);
		fu = selectedf.mul(dep_mask);
	}
	else
	    fu = selectedf;

	preprocess(fu);//Ԥ�������֡

	fu.convertTo(fu, CV_32F);//ͨ������һ������ת�������Ǿ���Ԫ�ص��������ͱ���һ����Ĭ�ϵ�fuΪCV_8U������Ҫת��Ϊ32λ�����ȸ�����
	dft(fu, Fu, DFT_COMPLEX_OUTPUT);

	Point deltaxy;//corelation���ҵ��������ĵ�仯��ֵ���������ĵ�������ͼ����������
	corelation(Fu, deltaxy);//ʹ��deltaxy�����������仯

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
		newselectarea.x += deltaxy.x;//ʹ��deltaxy�����������仯��������newselectarea�����Ͻ�����
		newselectarea.y += deltaxy.y;
		center.x += deltaxy.x;
		center.y += deltaxy.y;

		Mat newroi;
		getRectSubPix(frame, newselectarea.size(), center, newroi);//���ݸ��º�����꣬���¿�ѡ��������

		if (newroi.channels() != 1)
			cvtColor(newroi, newroi, COLOR_BGR2GRAY);
		preprocess(newroi);

		Mat F, A_new, B_new;

		//mulSpectrums(Hu, Fu, G, 0, true);//G�Ѿ���corelation����������
		//GΪ��һ֡���˲���H����һ֡������ͼ��F����Ӧ

		dft(newroi, F, DFT_COMPLEX_OUTPUT);
		mulSpectrums(G, F, A_new, 0, true);//�����G����newroi����һ֡��F����һ֡��H����Ӧ������һ֡��F����һ֡��H����Ӧ
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
VideoWriter writer("ForMat//BlurCar4//video.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, size); //�˴��ĳߴ���Բ�ͬ����Ƶ��Ҫ����																		  //��2��ѭ����ʾÿһ֡
int i = 0;
char name[2000];
//  Mat frame;//����һ��Mat���������ڴ洢ÿһ֡��ͼ��
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

	//������������������ı���
	ofstream OutFile("Output.txt");
	//ofstream FailureOut("Basketball//FailureOutput.txt");

	//��ȡ��Ƶ
	VideoCapture video("r_bear.avi");
	//car2,car4,carscale,fish,man,david,panda,woman
	Mat originframe;
	//Mat rotatef;
	Mat grayf;

	//��ʾ��һ֡ͼ��
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

	//�ڵ�һ֡ͼ���п�ѡĿ������
	newselectarea = selectROI("graywindow", grayf, true, false);
	//Ϊ����OTB2013�����ݼ����Ƚϣ����ֶ������һ֡��ROI����
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

	//cv::imshow("ROI", selectedf);//��ʾ��һ֡�е�ROI

	Mat depthframe1,depthframe;
	depthframe1 = imread("d_bear/1.jpg");
	cvtColor(depthframe1, depthframe, COLOR_BGR2GRAY);

	//cout << grayf.channels() << endl;
	//cout << depthframe.channels() << endl;
	
	//imshow("inidepframe", depthframe);
	//Size depthframeS = depthframe.size();
	char depthname[30];

	//���ݿ�ѡ����͵�һ֡ͼ�񣬳�ʼ���˲���
	ini_filter(grayf, newselectarea,depthframe);

	//��׽��꣬��������ƶ��켣��ʹ��onmouse�����õ��ں���֡�л������ο�����Ĳ���
	//setMouseCallback("graywindow", onmouse, 0);

	//ʱ�Ӻ�����ȡ����ʱ��
	//clock_t begin, end;
	//begin = clock();
	clock_t extrabegin, extraend;
	double extraruntime = 0;

	//��ȡ��Ƶ������ѭ���ṹ��ʾÿһ֡
	int i = 0;
	while (1) {

		i++;
		cout << i << endl;
		//cout << "PSR = " << PSR << endl;

		sprintf(depthname, "d_bear/%d.jpg", i);
		depthframe1 = imread(depthname);
		cvtColor(depthframe1, depthframe, COLOR_BGR2GRAY);
		//imshow("inidepframe", depthframe);

		//��Ƶ����frame����
		video >> originframe;

		//��ȡ������˳�
		if (originframe.empty() == true)
			break;

		//������ʾ������Ƶ���µ�һ֡
		cv::imshow("originwindow", originframe);

		//����������IpIImage��ʽ�µ����Ͻ�������תͼ��IpIImage* originf=originframe; originf->origin = 0;
		//����transpose()��flip()��תͼ��
		//transpose()��ͼ�����ת�ã�flip()�ɽ�����x��/y��/ԭ�㷭ת
		//transpose(originframe, rotatef);
		//flip(rotatef, rotatef, 1);
		//rotatef = originframe;

		//����cvtColor����ת��Ϊ�Ҷ�ͼ
		//COLOR_BGR2GRAYΪ�Ҷ�ͼ��ɫ�ʿռ�ת��ģʽ��code��
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

			//�½����ڽ���ʾ��ѡ����
			Mat selectedfforshow;
			getRectSubPix(grayf, newselectarea.size(), center, selectedfforshow, -1);

			//��graywindow������ʾ��׷�ٿ�ĻҶ�ͼ
			rectangle(grayf, newselectarea, Scalar(255, 255, 0), 2, 8, 0);
			cv::imshow("graywindow", grayf);
			cv::moveWindow("graywindow", 1000, 0);

		
		//����fps��ʾ��Ƶ
		int key = cv::waitKey(25);

		extraruntime += double((1000 * (extraend - extrabegin) / CLOCKS_PER_SEC));
		//cout << "the frame" << " " << i << " " << "is finished!" << endl;
		//����esc�˳���ʾ����
		//if (key == 27)
		//	break;
	}

	OutFile.close();

	//�������ʱ��
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
