//程序功能：Demo-V1.2-立体匹配与测距,输出深度图

//需要修改的参数：程序中共计5个位置需要根据应用场景来修改参数（已在代码旁做了备注）

//注意：第5处需要修改的地方，用来选择是运行SGBM或者运行BM算法，二者选一运行即可。



#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <opencv2/core.hpp>  

//#include"StereoSGBMParams_set.h"
#include <algorithm>
#include "opencv2/core/utility.hpp"
//#include "precomp.hpp"
#include <limits.h>
#include "opencv2/core/hal/intrin.hpp"
//#include "opencv2/core/internal.hpp"


using namespace std;
using namespace cv;
//using namespace mycv;

typedef uchar PixType;
typedef short CostType;
typedef short DispType;


static const int DEFAULT_RIGHT_BORDER = -1;

enum { NR = 16, NR2 = NR / 2 };



// the number of stripes is fixed, disregarding the number of threads/processors
// to make the results fully reproducible:


const int imageWidth  = 640;                      //摄像头单目的分辨率########--【需要调整参数的位置1】--#############
const int imageHeight = 480;

Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;                                   //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;                   //映射表  
Mat Rl, Rr, Pl, Pr, Q;                            //校正旋转矩阵R，投影矩阵P, 重投影矩阵Q
Mat xyz;                                          //三维坐标

Point origin;                                     //鼠标按下的起始点
Rect selection;                                   //定义矩形选框
bool selectObject = false;                        //是否选择对象




Ptr<StereoBM> bm = StereoBM::create(16, 9);


//########--【以下双目的标定参数为：需要调整参数的位置2】--#############
//相机双目标定的结果与如下各参数的对应关系见：双目标定结果说明.pdf，pdf文档位于main.cpp（即本文档）同级文件夹--#############

/*左目相机标定参数------------------------
fc_left_x   0            cc_left_x
0           fc_left_y    cc_left_y
0           0            1
-----------------------------------------*/

Mat cameraMatrixL = (Mat_<double>(3, 3) << 785.5112, 0, 300.8898,
	                                       0,          786.2416, 246.5716,
	                                       0,          0,            1);


Mat distCoeffL = (Mat_<double>(5, 1) << -0.3901, 0.153, -0.0015, 0.0018, 0.1086);
                                     //[kc_left_01,  kc_left_02,  kc_left_03,  kc_left_04,   kc_left_05]


/*右目相机标定参数------------------------
fc_right_x   0              cc_right_x
0            fc_right_y     cc_right_y
0            0              1
-----------------------------------------*/
Mat cameraMatrixR = (Mat_<double>(3, 3) << 782.7466, 0,         317.6625,
	                                       0,          783.0385, 240.6014,
	                                       0,          0,          1);


Mat distCoeffR = (Mat_<double>(5, 1) << -0.391,      -0.00087,    -0.0035,  -0.0023,0.547);
                                     //[kc_right_01,  kc_right_02,  kc_right_03,  kc_right_04,   kc_right_05]


Mat T = (Mat_<double>(3, 1) << -156.9611,  -0.7644,   -1.5846);    //T平移向量
							 //[T_01,        T_02,       T_03]

Mat rec = (Mat_<double>(3, 1) << 0.002, 0.0181, -0.0078);//-0.0099, 0.0048, 0); -0.01688, -0.00781, -0.00766);   //rec旋转向量
							  //[rec_01,     rec_02,     rec_03]

//########--双目的标定参数填写完毕-----------------------------------------------------------------------


Mat R;                                                     //R矩阵，用于中间计算

//--保存计算所得景深--------------------------------------------------
static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 16.0e4;
	//FILE* fp = fopen(filename, "wt");
	FILE *fp;
	fopen_s(&fp, filename, "w");
	if (!fp) {
		cout << "Cannot open file " << filename << endl;
		return;
	}
	printf("%d %d \n", mat.rows, mat.cols);
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);

		}
	}
	fclose(fp);
}

//--计算所得景深写入文件--------------------------------------------------
void writeMatToFile(cv::Mat& m, const char* filename) 
{
	ofstream File(filename, ios::out | ios::binary);
	//FILE *fp;
	//fopen_s(&fp, filename, "w");
	for (int i = 0; i<m.rows; i++)
	{ 
		for (int j = 0; j<m.cols; j++) 
		{ 
			//fprintf(fp, "%f", m.at<uchar>(i, j));
			uchar val = m.at<uchar>(i, j);
			//File<< (int(val) > 20 ? int(val) : 0);
			File<<int(val)<< endl ;
			File << ' ' << endl;
		}		
		//fprintf(fp, "\n");
		File<< endl;
	} 	
	//fclose(fp);
	File.close();
}

//--------------------------------------------------------------------------------------------------------
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)                             //颜色变换
{ 
	float max_val = 255.0f;
	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
	float sum = 0;
	for (int i = 0; i<8; i++)
		sum += map[i][3];

	float weights[8];   
	float cumsum[8];  
	cumsum[0] = 0;
	for (int i = 0; i<7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	int height_ = src.rows;
	int width_ = src.cols;
 
	for (int v = 0; v<height_; v++) {
		for (int u = 0; u<width_; u++) {
 
			float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

			int i;
			for (i = 0; i<7; i++)
				if (val<cumsum[i + 1])
					break;
 
			float   w = 1.0 - (val - cumsum[i])*weights[i];
			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
			 
			disp.data[v*width_ * 3 + 3 * u + 0] = b;                               //rgb内存连续存放 
			disp.data[v*width_ * 3 + 3 * u + 1] = g;
			disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}


//--描述：鼠标操作回调--------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:             //鼠标左按钮按下的事件
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		break;
	case EVENT_LBUTTONUP:               //鼠标左按钮释放的事件
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}


//--BM算法立体匹配--------------------------------------------------------------------
void stereo_match_bm(int, void*)
{
	int blockSize =8, uniquenessRatio = 5,  numDisparities = 6; //BM算法相关的参数，【需要调整参数的位置3，仅用于BM算法有效】--############

	bm->setBlockSize(2 * blockSize + 5);                           //SAD窗口大小，5~21之间为宜
	bm->setROI1(validROIL);                                        //左右视图的有效像素区域
	bm->setROI2(validROIR);
	bm->setPreFilterCap(61);                                       //预处理滤波器值
	bm->setMinDisparity(2);                                       //最小视差，默认值为0, 可以是负值，int型！！最短的测量距离
	bm->setNumDisparities(numDisparities * 16 );                   //视差窗口，即最大视差值与最小视差值之差,16的整数倍！！测距范围
	bm->setTextureThreshold(1000);
	bm->setUniquenessRatio(uniquenessRatio);                       //视差唯一性百分比,uniquenessRatio主要可以防止误匹配
	bm->setSpeckleWindowSize(100);                                 //检查视差连通区域变化度的窗口大小
	bm->setSpeckleRange(32);                                       //32视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	bm->setDisp12MaxDiff(-1);
	Mat disp;
	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);               //显示  
	bm->compute(rectifyImageL, rectifyImageR, disp);               //输入图像必须为灰度图

	reprojectImageTo3D(disp, xyz, Q, true);                        //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)
	xyz = xyz * 16;
	//saveXYZ("xyz.xls", xyz);

	disp.convertTo(disp, CV_32F, 1.0 / 16);                        //除以16得到真实视差值,disp.convertTo(disp, CV_32F, 1.0 );
	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

	medianBlur(disp8U, disp8U, 9);  //中值滤波

	Mat dispcolor(disp8U.size(), CV_8UC3);
	GenerateFalseMap(disp8U, dispcolor);
	//namedWindow("disparity", WINDOW_NORMAL);
	
	imshow("disparity", dispcolor);
	setMouseCallback("disparity", onMouse, 0);
	
}


//--经过修改的calcPixelCostBT函数--------------------------------------------------------------------
void calcPixelCostBT(const Mat& img1, const Mat& img2, int y,
	int minD, int maxD, CostType* cost,
	PixType* buffer, const PixType* tab,
	int tabOfs, int, int xrange_min = 0, int xrange_max = DEFAULT_RIGHT_BORDER)
{
	int x, c, width = img1.cols, cn = img1.channels();
	int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
	int D = maxD - minD, width1 = maxX1 - minX1;
	//This minX1 & maxX2 correction is defining which part of calculatable line must be calculated
	//That is needs of parallel algorithm
	xrange_min = (xrange_min < 0) ? 0 : xrange_min;
	xrange_max = (xrange_max == DEFAULT_RIGHT_BORDER) || (xrange_max > width1) ? width1 : xrange_max;
	maxX1 = minX1 + xrange_max;
	minX1 += xrange_min;
	width1 = maxX1 - minX1;
	int minX2 = std::max(minX1 - maxD, 0), maxX2 = std::min(maxX1 - minD, width);
	int width2 = maxX2 - minX2;
	const PixType *row1 = img1.ptr<PixType>(y), *row2 = img2.ptr<PixType>(y);
	PixType *prow1 = buffer + width2 * 2, *prow2 = prow1 + width * cn * 2;
#if CV_SIMD128

	bool useSIMD = hasSIMD128();
		//hasSIMD128();
#endif

	tab += tabOfs;

	for (c = 0; c < cn * 2; c++)
	{
		prow1[width*c] = prow1[width*c + width - 1] =
			prow2[width*c] = prow2[width*c + width - 1] = tab[0];
	}

	int n1 = y > 0 ? -(int)img1.step : 0, s1 = y < img1.rows - 1 ? (int)img1.step : 0;
	int n2 = y > 0 ? -(int)img2.step : 0, s2 = y < img2.rows - 1 ? (int)img2.step : 0;

	int minX_cmn = std::min(minX1, minX2) - 1;
	int maxX_cmn = std::max(maxX1, maxX2) + 1;
	minX_cmn = std::max(minX_cmn, 1);
	maxX_cmn = std::min(maxX_cmn, width - 1);
	if (cn == 1)
	{
		for (x = minX_cmn; x < maxX_cmn; x++)
		{
			prow1[x] = tab[(row1[x + 1] - row1[x - 1]) * 2 + row1[x + n1 + 1] - row1[x + n1 - 1] + row1[x + s1 + 1] - row1[x + s1 - 1]];
			prow2[width - 1 - x] = tab[(row2[x + 1] - row2[x - 1]) * 2 + row2[x + n2 + 1] - row2[x + n2 - 1] + row2[x + s2 + 1] - row2[x + s2 - 1]];

			prow1[x + width] = row1[x];
			prow2[width - 1 - x + width] = row2[x];
		}
	}
	else
	{
		for (x = minX_cmn; x < maxX_cmn; x++)
		{
			prow1[x] = tab[(row1[x * 3 + 3] - row1[x * 3 - 3]) * 2 + row1[x * 3 + n1 + 3] - row1[x * 3 + n1 - 3] + row1[x * 3 + s1 + 3] - row1[x * 3 + s1 - 3]];
			prow1[x + width] = tab[(row1[x * 3 + 4] - row1[x * 3 - 2]) * 2 + row1[x * 3 + n1 + 4] - row1[x * 3 + n1 - 2] + row1[x * 3 + s1 + 4] - row1[x * 3 + s1 - 2]];
			prow1[x + width * 2] = tab[(row1[x * 3 + 5] - row1[x * 3 - 1]) * 2 + row1[x * 3 + n1 + 5] - row1[x * 3 + n1 - 1] + row1[x * 3 + s1 + 5] - row1[x * 3 + s1 - 1]];

			prow2[width - 1 - x] = tab[(row2[x * 3 + 3] - row2[x * 3 - 3]) * 2 + row2[x * 3 + n2 + 3] - row2[x * 3 + n2 - 3] + row2[x * 3 + s2 + 3] - row2[x * 3 + s2 - 3]];
			prow2[width - 1 - x + width] = tab[(row2[x * 3 + 4] - row2[x * 3 - 2]) * 2 + row2[x * 3 + n2 + 4] - row2[x * 3 + n2 - 2] + row2[x * 3 + s2 + 4] - row2[x * 3 + s2 - 2]];
			prow2[width - 1 - x + width * 2] = tab[(row2[x * 3 + 5] - row2[x * 3 - 1]) * 2 + row2[x * 3 + n2 + 5] - row2[x * 3 + n2 - 1] + row2[x * 3 + s2 + 5] - row2[x * 3 + s2 - 1]];

			prow1[x + width * 3] = row1[x * 3];
			prow1[x + width * 4] = row1[x * 3 + 1];
			prow1[x + width * 5] = row1[x * 3 + 2];

			prow2[width - 1 - x + width * 3] = row2[x * 3];
			prow2[width - 1 - x + width * 4] = row2[x * 3 + 1];
			prow2[width - 1 - x + width * 5] = row2[x * 3 + 2];
		}
	}

	memset(cost + xrange_min * D, 0, width1*D * sizeof(cost[0]));

	buffer -= width - 1 - maxX2;
	cost -= (minX1 - xrange_min)*D + minD; // simplify the cost indices inside the loop

	for (c = 0; c < cn * 2; c++, prow1 += width, prow2 += width)
	{
		int diff_scale = c < cn ? 0 : 2;

		// precompute
		//   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
		//   v1 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
		for (x = width - 1 - maxX2; x < width - 1 - minX2; x++)
		{
			int v = prow2[x];
			int vl = x > 0 ? (v + prow2[x - 1]) / 2 : v;
			int vr = x < width - 1 ? (v + prow2[x + 1]) / 2 : v;
			int v0 = std::min(vl, vr); v0 = std::min(v0, v);
			int v1 = std::max(vl, vr); v1 = std::max(v1, v);
			buffer[x] = (PixType)v0;
			buffer[x + width2] = (PixType)v1;
		}

		for (x = minX1; x < maxX1; x++)
		{
			int u = prow1[x];
			int ul = x > 0 ? (u + prow1[x - 1]) / 2 : u;
			int ur = x < width - 1 ? (u + prow1[x + 1]) / 2 : u;
			int u0 = std::min(ul, ur); u0 = std::min(u0, u);
			int u1 = std::max(ul, ur); u1 = std::max(u1, u);

#if CV_SIMD128
			if (useSIMD)
			{
				v_uint8x16 _u = v_setall_u8((uchar)u), _u0 = v_setall_u8((uchar)u0);
				v_uint8x16 _u1 = v_setall_u8((uchar)u1);

				for (int d = minD; d < maxD; d += 16)
				{
					v_uint8x16 _v = v_load(prow2 + width - x - 1 + d);
					v_uint8x16 _v0 = v_load(buffer + width - x - 1 + d);
					v_uint8x16 _v1 = v_load(buffer + width - x - 1 + d + width2);
					v_uint8x16 c0 = v_max(_u - _v1, _v0 - _u);
					v_uint8x16 c1 = v_max(_v - _u1, _u0 - _v);
					v_uint8x16 diff = v_min(c0, c1);

					v_int16x8 _c0 = v_load_aligned(cost + x * D + d);
					v_int16x8 _c1 = v_load_aligned(cost + x * D + d + 8);

					v_uint16x8 diff1, diff2;
					v_expand(diff, diff1, diff2);
					v_store_aligned(cost + x * D + d, _c0 + v_reinterpret_as_s16(diff1 >> diff_scale));
					v_store_aligned(cost + x * D + d + 8, _c1 + v_reinterpret_as_s16(diff2 >> diff_scale));
				}
			}
			else
#endif
			{
				for (int d = minD; d < maxD; d++)
				{
					int v = prow2[width - x - 1 + d];
					int v0 = buffer[width - x - 1 + d];
					int v1 = buffer[width - x - 1 + d + width2];
					int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
					int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

					cost[x*D + d] = (CostType)(cost[x*D + d] + (std::min(c0, c1) >> diff_scale));
				}
			}
		}
	}
}

//--经过修改的SGBM函数--------------------------------------------------------------------
void computeDisparitySGBM(const Mat& img1, const Mat& img2,
	Mat& disp1,
	//const StereoSGBMParams1& params, 
	int minDisparity, int numDisparities, int SADWindowSize, int preFilterCap, int uniquenessRatio, int P1, int P2, int speckleWindowSize, int speckleRange, int disp12MaxDiff, int mode,
	Mat& buffer)
{

#if CV_SIMD128
	volatile bool useSIMD = hasSIMD128();  //获取CPU指令集优化状态 
	//CV_SIMD128
#endif

#if CV_SIMD128
	// maxDisparity is supposed to multiple of 16, so we can forget doing else
	static const uchar LSBTab[] =
	{
		0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
		5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
	};
	static const v_uint16x8 v_LSB = v_uint16x8(0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);

	
#endif

	const int ALIGN = 16;
	const int DISP_SHIFT = StereoMatcher::DISP_SHIFT;
	const int DISP_SCALE = (1 << DISP_SHIFT);
	const CostType MAX_COST = SHRT_MAX;

	/*
	int minD = params.minDisparity, maxD = minD + params.numDisparities;
	Size SADWindowSize;
	SADWindowSize.width = SADWindowSize.height = params.SADWindowSize > 0 ? params.SADWindowSize : 5;
	int ftzero = std::max(params.preFilterCap, 15) | 1;
	int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
	int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
	int P1 = params.P1 > 0 ? params.P1 : 2, P2 = std::max(params.P2 > 0 ? params.P2 : 5, P1 + 1);
	int k, width = disp1.cols, height = disp1.rows;
	int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
	int D = maxD - minD, width1 = maxX1 - minX1;
	int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
	int SW2 = SADWindowSize.width / 2, SH2 = SADWindowSize.height / 2;
	bool fullDP = params.mode == StereoSGBM::MODE_HH;
	int npasses = fullDP ? 2 : 1;
	const int TAB_OFS = 256 * 4, TAB_SIZE = 256 + TAB_OFS * 2;
	PixType clipTab[TAB_SIZE];
	*/

	int minD = minDisparity;
	int maxD = minD + numDisparities;
	Size SADWindowSize1;
	SADWindowSize1.width = SADWindowSize1.height = SADWindowSize > 0 ? SADWindowSize : 5;
	int ftzero = std::max(preFilterCap, 15) | 1;
	uniquenessRatio = uniquenessRatio >= 0 ? uniquenessRatio : 10;
	disp12MaxDiff = disp12MaxDiff > 0 ? disp12MaxDiff : 1;
	P1 = P1 > 0 ? P1 : 2, P2 = std::max(P2 > 0 ? P2 : 5, P1 + 1);
	int k, width = disp1.cols, height = disp1.rows;
	
	int minX1 = std::max(maxD, 0), maxX1 = width + std::min(minD, 0);
	int D = maxD - minD, width1 = maxX1 - minX1;
	int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP * DISP_SCALE;
	int SW2 = SADWindowSize1.width / 2, SH2 = SADWindowSize1.height / 2;
	bool fullDP = mode == StereoSGBM::MODE_HH;
	int npasses = fullDP ? 2 : 1;
	const int TAB_OFS = 256 * 4, TAB_SIZE = 256 + TAB_OFS * 2;
	PixType clipTab[TAB_SIZE];

	

	for (k = 0; k < TAB_SIZE; k++)
		clipTab[k] = (PixType)(std::min(std::max(k - TAB_OFS, -ftzero), ftzero) + ftzero);
	
	if (minX1 >= maxX1)
	{
		disp1 = Scalar::all(INVALID_DISP_SCALED);
		return;
		
	}
	
	CV_Assert(D % 16 == 0);
	
	// NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
	// if you change NR, please, modify the loop as well.
	int D2 = D + 16, NRD2 = NR2 * D2;

	// the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
	// for 8-way dynamic programming we need the current row and
	// the previous row, i.e. 2 rows in total
	const int NLR = 2;
	const int LrBorder = NLR - 1;
	
	// for each possible stereo match (img1(x,y) <=> img2(x-d,y))
	// we keep pixel difference cost (C) and the summary cost over NR directions (S).
	// we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
	size_t costBufSize = width1 * D;
	size_t CSBufSize = costBufSize * (fullDP ? height : 1);
	size_t minLrSize = (width1 + LrBorder * 2)*NR2, LrSize = minLrSize * D2;
	int hsumBufNRows = SH2 * 2 + 2;
	size_t totalBufSize = (LrSize + minLrSize)*NLR * sizeof(CostType) + // minLr[] and Lr[]
		costBufSize * (hsumBufNRows + 1) * sizeof(CostType) + // hsumBuf, pixdiff
		CSBufSize * 2 * sizeof(CostType) + // C, S
		width * 16 * img1.channels() * sizeof(PixType) + // temp buffer for computing per-pixel cost
		width * (sizeof(CostType) + sizeof(DispType)) + 1024; // disp2cost + disp2
	
	if (buffer.empty() || !buffer.isContinuous() ||
		buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize)
		buffer.reserveBuffer(totalBufSize);
	// summary cost over different (nDirs) directions
	CostType* Cbuf = (CostType*)alignPtr(buffer.ptr(), ALIGN);
	CostType* Sbuf = Cbuf + CSBufSize;
	CostType* hsumBuf = Sbuf + CSBufSize;
	CostType* pixDiff = hsumBuf + costBufSize * hsumBufNRows;

	CostType* disp2cost = pixDiff + costBufSize + (LrSize + minLrSize)*NLR;
	DispType* disp2ptr = (DispType*)(disp2cost + width);
	PixType* tempBuf = (PixType*)(disp2ptr + width);

	// add P2 to every C(x,y). it saves a few operations in the inner loops
	for (k = 0; k < (int)CSBufSize; k++)
		Cbuf[k] = (CostType)P2;
	
	for (int pass = 1; pass <= npasses; pass++)
	{
		int x1, y1, x2, y2, dx, dy;

		if (pass == 1)
		{
			y1 = 0; y2 = height; dy = 1;
			x1 = 0; x2 = width1; dx = 1;
		}
		else
		{
			y1 = height - 1; y2 = -1; dy = -1;
			x1 = width1 - 1; x2 = -1; dx = -1;
		}

		CostType *Lr[NLR] = { 0 }, *minLr[NLR] = { 0 };

		for (k = 0; k < NLR; k++)
		{
			// shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
			// and will occasionally use negative indices with the arrays
			// we need to shift Lr[k] pointers by 1, to give the space for d=-1.
			// however, then the alignment will be imperfect, i.e. bad for SSE,
			// thus we shift the pointers by 8 (8*sizeof(short) == 16 - ideal alignment)
			Lr[k] = pixDiff + costBufSize + LrSize * k + NRD2 * LrBorder + 8;
			memset(Lr[k] - LrBorder * NRD2 - 8, 0, LrSize * sizeof(CostType));
			minLr[k] = pixDiff + costBufSize + LrSize * NLR + minLrSize * k + NR2 * LrBorder;
			memset(minLr[k] - LrBorder * NR2, 0, minLrSize * sizeof(CostType));
		}

		for (int y = y1; y != y2; y += dy)
		{
			int x, d;
			DispType* disp1ptr = disp1.ptr<DispType>(y);
			CostType* C = Cbuf + (!fullDP ? 0 : y * costBufSize);
			CostType* S = Sbuf + (!fullDP ? 0 : y * costBufSize);

			if (pass == 1) // compute C on the first pass, and reuse it on the second pass, if any.
			{
				int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

				for (k = dy1; k <= dy2; k++)
				{
					CostType* hsumAdd = hsumBuf + (std::min(k, height - 1) % hsumBufNRows)*costBufSize;

					if (k < height)
					{
						calcPixelCostBT(img1, img2, k, minD, maxD, pixDiff, tempBuf, clipTab, TAB_OFS, ftzero);
						
						
						memset(hsumAdd, 0, D * sizeof(CostType));
						for (x = 0; x <= SW2 * D; x += D)
						{
							int scale = x == 0 ? SW2 + 1 : 1;
							for (d = 0; d < D; d++)
								hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d] * scale);
						}

						if (y > 0)
						{
							const CostType* hsumSub = hsumBuf + (std::max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
							const CostType* Cprev = !fullDP || y == 0 ? C : C - costBufSize;

							for (x = D; x < width1*D; x += D)
							{
								const CostType* pixAdd = pixDiff + std::min(x + SW2 * D, (width1 - 1)*D);
								const CostType* pixSub = pixDiff + std::max(x - (SW2 + 1)*D, 0);

#if CV_SIMD128
								if (useSIMD)
								{
									for (d = 0; d < D; d += 8)
									{
										v_int16x8 hv = v_load(hsumAdd + x - D + d);
										v_int16x8 Cx = v_load(Cprev + x + d);
										v_int16x8 psub = v_load(pixSub + d);
										v_int16x8 padd = v_load(pixAdd + d);
										hv = (hv - psub + padd);
										psub = v_load(hsumSub + x + d);
										Cx = Cx - psub + hv;
										v_store(hsumAdd + x + d, hv);
										v_store(C + x + d, Cx);
									}
								}
								else
#endif
								{
									for (d = 0; d < D; d++)
									{
										int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
										C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
									}
								}
							}
						}
						else
						{
							for (x = D; x < width1*D; x += D)
							{
								const CostType* pixAdd = pixDiff + std::min(x + SW2 * D, (width1 - 1)*D);
								const CostType* pixSub = pixDiff + std::max(x - (SW2 + 1)*D, 0);

								for (d = 0; d < D; d++)
									hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
							}
						}
					}
					
					if (y == 0)
					{
						int scale = k == 0 ? SH2 + 1 : 1;
						for (x = 0; x < width1*D; x++)
							C[x] = (CostType)(C[x] + hsumAdd[x] * scale);
					}
				}
				
				// also, clear the S buffer
				for (k = 0; k < width1*D; k++)
					S[k] = 0;
			}
			
			// clear the left and the right borders
			memset(Lr[0] - NRD2 * LrBorder - 8, 0, NRD2*LrBorder * sizeof(CostType));
			memset(Lr[0] + width1 * NRD2 - 8, 0, NRD2*LrBorder * sizeof(CostType));
			memset(minLr[0] - NR2 * LrBorder, 0, NR2*LrBorder * sizeof(CostType));
			memset(minLr[0] + width1 * NR2, 0, NR2*LrBorder * sizeof(CostType));

			/*
			[formula 13 in the paper]
			compute L_r(p, d) = C(p, d) +
			min(L_r(p-r, d),
			L_r(p-r, d-1) + P1,
			L_r(p-r, d+1) + P1,
			min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
			where p = (x,y), r is one of the directions.
			we process all the directions at once:
			0: r=(-dx, 0)
			1: r=(-1, -dy)
			2: r=(0, -dy)
			3: r=(1, -dy)
			4: r=(-2, -dy)
			5: r=(-1, -dy*2)
			6: r=(1, -dy*2)
			7: r=(2, -dy)
			*/

			for (x = x1; x != x2; x += dx)
			{
				int xm = x * NR2, xd = xm * D2;

				int delta0 = minLr[0][xm - dx * NR2] + P2, delta1 = minLr[1][xm - NR2 + 1] + P2;
				int delta2 = minLr[1][xm + 2] + P2, delta3 = minLr[1][xm + NR2 + 3] + P2;

				CostType* Lr_p0 = Lr[0] + xd - dx * NRD2;
				CostType* Lr_p1 = Lr[1] + xd - NRD2 + D2;
				CostType* Lr_p2 = Lr[1] + xd + D2 * 2;
				CostType* Lr_p3 = Lr[1] + xd + NRD2 + D2 * 3;

				Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] =
					Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;

				CostType* Lr_p = Lr[0] + xd;
				const CostType* Cp = C + x * D;
				CostType* Sp = S + x * D;

#if CV_SIMD128
				if (useSIMD)
				{
					v_int16x8 _P1 = v_setall_s16((short)P1);

					v_int16x8 _delta0 = v_setall_s16((short)delta0);
					v_int16x8 _delta1 = v_setall_s16((short)delta1);
					v_int16x8 _delta2 = v_setall_s16((short)delta2);
					v_int16x8 _delta3 = v_setall_s16((short)delta3);
					v_int16x8 _minL0 = v_setall_s16((short)MAX_COST);

					for (d = 0; d < D; d += 8)
					{
						v_int16x8 Cpd = v_load(Cp + d);
						v_int16x8 L0, L1, L2, L3;

						L0 = v_load(Lr_p0 + d);
						L1 = v_load(Lr_p1 + d);
						L2 = v_load(Lr_p2 + d);
						L3 = v_load(Lr_p3 + d);

						L0 = v_min(L0, (v_load(Lr_p0 + d - 1) + _P1));
						L0 = v_min(L0, (v_load(Lr_p0 + d + 1) + _P1));

						L1 = v_min(L1, (v_load(Lr_p1 + d - 1) + _P1));
						L1 = v_min(L1, (v_load(Lr_p1 + d + 1) + _P1));

						L2 = v_min(L2, (v_load(Lr_p2 + d - 1) + _P1));
						L2 = v_min(L2, (v_load(Lr_p2 + d + 1) + _P1));

						L3 = v_min(L3, (v_load(Lr_p3 + d - 1) + _P1));
						L3 = v_min(L3, (v_load(Lr_p3 + d + 1) + _P1));

						L0 = v_min(L0, _delta0);
						L0 = ((L0 - _delta0) + Cpd);

						L1 = v_min(L1, _delta1);
						L1 = ((L1 - _delta1) + Cpd);

						L2 = v_min(L2, _delta2);
						L2 = ((L2 - _delta2) + Cpd);

						L3 = v_min(L3, _delta3);
						L3 = ((L3 - _delta3) + Cpd);

						v_store(Lr_p + d, L0);
						v_store(Lr_p + d + D2, L1);
						v_store(Lr_p + d + D2 * 2, L2);
						v_store(Lr_p + d + D2 * 3, L3);

						// Get minimum from in L0-L3
						v_int16x8 t02L, t02H, t13L, t13H, t0123L, t0123H;
						v_zip(L0, L2, t02L, t02H);            // L0[0] L2[0] L0[1] L2[1]...
						v_zip(L1, L3, t13L, t13H);            // L1[0] L3[0] L1[1] L3[1]...
						v_int16x8 t02 = v_min(t02L, t02H);    // L0[i] L2[i] L0[i] L2[i]...
						v_int16x8 t13 = v_min(t13L, t13H);    // L1[i] L3[i] L1[i] L3[i]...
						v_zip(t02, t13, t0123L, t0123H);      // L0[i] L1[i] L2[i] L3[i]...
						v_int16x8 t0 = v_min(t0123L, t0123H);
						_minL0 = v_min(_minL0, t0);

						v_int16x8 Sval = v_load(Sp + d);

						L0 = L0 + L1;
						L2 = L2 + L3;
						Sval = Sval + L0;
						Sval = Sval + L2;

						v_store(Sp + d, Sval);
					}

					v_int32x4 minL, minH;
					v_expand(_minL0, minL, minH);
					v_pack_store(&minLr[0][xm], v_min(minL, minH));
				}
				else
#endif
				{
					int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

					for (d = 0; d < D; d++)
					{
						int Cpd = Cp[d], L0, L1, L2, L3;

						L0 = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) - delta0;
						L1 = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d - 1] + P1, std::min(Lr_p1[d + 1] + P1, delta1))) - delta1;
						L2 = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d - 1] + P1, std::min(Lr_p2[d + 1] + P1, delta2))) - delta2;
						L3 = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d - 1] + P1, std::min(Lr_p3[d + 1] + P1, delta3))) - delta3;

						Lr_p[d] = (CostType)L0;
						minL0 = std::min(minL0, L0);

						Lr_p[d + D2] = (CostType)L1;
						minL1 = std::min(minL1, L1);

						Lr_p[d + D2 * 2] = (CostType)L2;
						minL2 = std::min(minL2, L2);

						Lr_p[d + D2 * 3] = (CostType)L3;
						minL3 = std::min(minL3, L3);

						Sp[d] = saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
					}
					minLr[0][xm] = (CostType)minL0;
					minLr[0][xm + 1] = (CostType)minL1;
					minLr[0][xm + 2] = (CostType)minL2;
					minLr[0][xm + 3] = (CostType)minL3;
				}
			}
			
			if (pass == npasses)
			{
				for (x = 0; x < width; x++)
				{
					 
					disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
					disp2cost[x] = MAX_COST;
				}

				for (x = width1 - 1; x >= 0; x--)
				{
					 
					CostType* Sp = S + x * D;
					int minS = MAX_COST, bestDisp = -1;

					if (npasses == 1)
					{
						int xm = x * NR2, xd = xm * D2;

						int minL0 = MAX_COST;
						int delta0 = minLr[0][xm + NR2] + P2;
						CostType* Lr_p0 = Lr[0] + xd + NRD2;
						Lr_p0[-1] = Lr_p0[D] = MAX_COST;
						CostType* Lr_p = Lr[0] + xd;

						const CostType* Cp = C + x * D;

#if CV_SIMD128
						if (useSIMD)
						{
							v_int16x8 _P1 = v_setall_s16((short)P1);
							v_int16x8 _delta0 = v_setall_s16((short)delta0);

							v_int16x8 _minL0 = v_setall_s16((short)minL0);
							v_int16x8 _minS = v_setall_s16(MAX_COST), _bestDisp = v_setall_s16(-1);
							v_int16x8 _d8 = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = v_setall_s16(8);

							for (d = 0; d < D; d += 8)
							{
								v_int16x8 Cpd = v_load(Cp + d);
								v_int16x8 L0 = v_load(Lr_p0 + d);

								L0 = v_min(L0, v_load(Lr_p0 + d - 1) + _P1);
								L0 = v_min(L0, v_load(Lr_p0 + d + 1) + _P1);
								L0 = v_min(L0, _delta0);
								L0 = L0 - _delta0 + Cpd;

								v_store(Lr_p + d, L0);
								_minL0 = v_min(_minL0, L0);
								L0 = L0 + v_load(Sp + d);
								v_store(Sp + d, L0);

								v_int16x8 mask = _minS > L0;
								_minS = v_min(_minS, L0);
								_bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
								_d8 += _8;
							}
							short bestDispBuf[8];
							v_store(bestDispBuf, _bestDisp);

							v_int32x4 min32L, min32H;
							v_expand(_minL0, min32L, min32H);
							minLr[0][xm] = (CostType)std::min(v_reduce_min(min32L), v_reduce_min(min32H));

							v_expand(_minS, min32L, min32H);
							minS = std::min(v_reduce_min(min32L), v_reduce_min(min32H));

							v_int16x8 ss = v_setall_s16((short)minS);
							v_uint16x8 minMask = v_reinterpret_as_u16(ss == _minS);
							v_uint16x8 minBit = minMask & v_LSB;

							v_uint32x4 minBitL, minBitH;
							v_expand(minBit, minBitL, minBitH);

							int idx = v_reduce_sum(minBitL) + v_reduce_sum(minBitH);
							bestDisp = bestDispBuf[LSBTab[idx]];
						}
						else
#endif
						{
							for (d = 0; d < D; d++)
							{
								int L0 = Cp[d] + std::min((int)Lr_p0[d], std::min(Lr_p0[d - 1] + P1, std::min(Lr_p0[d + 1] + P1, delta0))) - delta0;

								Lr_p[d] = (CostType)L0;
								minL0 = std::min(minL0, L0);

								int Sval = Sp[d] = saturate_cast<CostType>(Sp[d] + L0);
								if (Sval < minS)
								{
									minS = Sval;
									bestDisp = d;
								}
							}
							minLr[0][xm] = (CostType)minL0;
						}
					}
					else
					{
#if CV_SIMD128
						if (useSIMD)
						{
							v_int16x8 _minS = v_setall_s16(MAX_COST), _bestDisp = v_setall_s16(-1);
							v_int16x8 _d8 = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), _8 = v_setall_s16(8);

							for (d = 0; d < D; d += 8)
							{
								v_int16x8 L0 = v_load(Sp + d);
								v_int16x8 mask = L0 < _minS;
								_minS = v_min(L0, _minS);
								_bestDisp = _bestDisp ^ ((_bestDisp ^ _d8) & mask);
								_d8 = _d8 + _8;
							}
							v_int32x4 _d0, _d1;
							v_expand(_minS, _d0, _d1);
							minS = (int)std::min(v_reduce_min(_d0), v_reduce_min(_d1));
							v_int16x8 v_mask = v_setall_s16((short)minS) == _minS;

							_bestDisp = (_bestDisp & v_mask) | (v_setall_s16(SHRT_MAX) & ~v_mask);
							v_expand(_bestDisp, _d0, _d1);
							bestDisp = (int)std::min(v_reduce_min(_d0), v_reduce_min(_d1));
						}
						else
#endif
						{
							for (d = 0; d < D; d++)
							{
								int Sval = Sp[d];
								if (Sval < minS)
								{
									minS = Sval;
									bestDisp = d;
								}
							}
						}
					}

					for (d = 0; d < D; d++)
					{
						if (Sp[d] * (100 - uniquenessRatio) < minS * 100 && std::abs(bestDisp - d) > 1)
							break;
					}
					if (d < D)
						continue;
					d = bestDisp;
					int _x2 = x + minX1 - d - minD;
					if (disp2cost[_x2] > minS)
					{
						disp2cost[_x2] = (CostType)minS;
						disp2ptr[_x2] = (DispType)(d + minD);
					}

					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int denom2 = std::max(Sp[d - 1] + Sp[d + 1] - 2 * Sp[d], 1);
						d = d * DISP_SCALE + ((Sp[d - 1] - Sp[d + 1])*DISP_SCALE + denom2) / (denom2 * 2);
					}
					else
						d *= DISP_SCALE;
					disp1ptr[x + minX1] = (DispType)(d + minD * DISP_SCALE);
				}

				for (x = minX1; x < maxX1; x++)
				{
					// we round the computed disparity both towards -inf and +inf and check
					// if either of the corresponding disparities in disp2 is consistent.
					// This is to give the computed disparity a chance to look valid if it is.
					int d1 = disp1ptr[x];
					 
					if (d1 == INVALID_DISP_SCALED)
						continue;
					int _d = d1 >> DISP_SHIFT;
					int d_ = (d1 + DISP_SCALE - 1) >> DISP_SHIFT;
					int _x = x - _d, x_ = x - d_;
					if (0 <= _x && _x < width && disp2ptr[_x] >= minD && std::abs(disp2ptr[_x] - _d) > disp12MaxDiff &&
						0 <= x_ && x_ < width && disp2ptr[x_] >= minD && std::abs(disp2ptr[x_] - d_) > disp12MaxDiff)
						disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
				}
			}
			// now shift the cyclic buffers
			std::swap(Lr[0], Lr[1]);
			std::swap(minLr[0], minLr[1]);
		}
	}
}


void stereo_match_sgbm(int, void*)                                         //SGBM匹配算法
{
	int mindisparity = 16;                                                 //最小视差
	int SADWindowSize = 8;                                                //滑动窗口的大小
	int ndisparities = 64;                                                //最大的视差，要被16整除
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);

	int P1 = 4 * rectifyImageL.channels() * SADWindowSize* SADWindowSize;  //惩罚系数1
	int P2 = 32 * rectifyImageL.channels() * SADWindowSize* SADWindowSize; //惩罚系数2
	sgbm->setP1(P1);
	sgbm->setP2(P2);

	sgbm->setPreFilterCap(60);                                             //滤波系数
	sgbm->setUniquenessRatio(30);                                          //代价方程概率因子
	sgbm->setSpeckleRange(2);                                              //相邻像素点的视差值浮动范围
	sgbm->setSpeckleWindowSize(200);                                       //针对散斑滤波的窗口大小
	sgbm->setDisp12MaxDiff(1);                                             //视差图的像素点检查
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);  

	Mat disp;
	sgbm->compute(rectifyImageL, rectifyImageR, disp);

	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);                       //显示  

	reprojectImageTo3D(disp, xyz, Q, true);                                //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)
	xyz = xyz * 16;

	disp.convertTo(disp, CV_32F, 1.0 / 16);                                //除以16得到真实视差值,disp.convertTo(disp, CV_32F, 1.0 );
	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

	medianBlur(disp8U, disp8U, 9);                                             //中值滤波

	Mat dispcolor(disp8U.size(), CV_8UC3);
	GenerateFalseMap(disp8U, dispcolor);

	imshow("disparity", dispcolor);
	setMouseCallback("disparity", onMouse, 0);
	
}


//--立体匹配，立体匹配函数参数设定，函数调用------------------------------------------------------
void stereo_match(int, void*)
{
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(63);
	int sgbmWinSize = 8;//根据实际情况自己设定
	int NumDisparities = 64;//根据实际情况自己设定
	int UniquenessRatio = 12;//根据实际情况自己设定
	sgbm->setBlockSize(sgbmWinSize);
	int cn = rectifyImageL.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(16);
	sgbm->setNumDisparities(NumDisparities);
	sgbm->setUniquenessRatio(UniquenessRatio);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(10);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);


	/*
	mycv::StereoSGBMParams1 StereoSGBMParams2 =StereoSGBMParams1(0, NumDisparities, 5,
		8 * cn*sgbmWinSize*sgbmWinSize, 32 * cn*sgbmWinSize*sgbmWinSize, 1, 0,
		6, 100, 10,
		//StereoSGBM::MODE_SGBM
		StereoSGBM::MODE_HH);
	StereoSGBMParams2.minDisparity = 0;
	StereoSGBMParams2.numDisparities = NumDisparities;
	StereoSGBMParams2.SADWindowSize=5;
	StereoSGBMParams2.preFilterCap=63;
	StereoSGBMParams2.uniquenessRatio= UniquenessRatio;
	StereoSGBMParams2.P1= 8 * cn*sgbmWinSize*sgbmWinSize;
	StereoSGBMParams2.P2= 32 * cn*sgbmWinSize*sgbmWinSize;
	StereoSGBMParams2.speckleWindowSize=100;
	StereoSGBMParams2.speckleRange=10;
	StereoSGBMParams2.disp12MaxDiff=1;
	StereoSGBMParams2.mode= StereoSGBM::MODE_HH;
	*/
	/*
	Mat disp, dispf, disp8;
	sgbm->compute(rectifyImageL, rectifyImageR, disp);
	//去黑边
	Mat img1p, img2p;
	copyMakeBorder(rectifyImageL, img1p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(rectifyImageR, img2p, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	dispf = disp.colRange(NumDisparities, img2p.cols - NumDisparities);
	dispf.convertTo(disp8, CV_8U, 255 / (NumDisparities *16.));
	
	reprojectImageTo3D(dispf, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity", disp8);
	Mat color(dispf.size(), CV_8UC3);
	GenerateFalseMap(disp8, color);//转成彩图
	imshow("disparity", color);
	setMouseCallback("disparity", onMouse, 0);
	cvWaitKey(10) == 27;
	saveXYZ("xyz.xls", xyz);
	*/
	Mat disp2, disp9, dispf2;
	Mat disparr;
	Mat left = rectifyImageL, right = rectifyImageR;
	CV_Assert(left.size() == right.size() && left.type() == right.type() &&
		left.depth() == CV_8U);


	disparr.create(left.size(), CV_16S);
	disp2 = disparr;
	//CV_INSTRUMENT_REGION()
	cv::Mat buffer;

	computeDisparitySGBM(rectifyImageL, rectifyImageR, disp2,
		//StereoSGBMParams2,
		8, NumDisparities, 5, 63, UniquenessRatio, 8 * cn*sgbmWinSize*sgbmWinSize, 32 * cn*sgbmWinSize*sgbmWinSize, 100, 10, 1, StereoSGBM::MODE_SGBM,
		buffer);

	//imwrite("dispcolor2L.jpg", disp2);
	Mat disp8U2= Mat(disp2.rows, disp2.cols, CV_8UC1);                       //显示  


	//reprojectImageTo3D(disp2, xyz, Q, true);                                //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)
	//xyz = xyz * 16;

	disp2.convertTo(disp2, CV_32F, 1.0 / 16);                                //除以16得到真实视差值,disp.convertTo(disp, CV_32F, 1.0 );
	normalize(disp2, disp8U2, 0, 255, NORM_MINMAX, CV_8UC1);

	medianBlur(disp8U2, disp8U2, 9);                                             //中值滤波

	Mat dispcolor2(disp8U2.size(), CV_8UC3);
	GenerateFalseMap(disp8U2, dispcolor2);
	string rgbImageL_calib_result = "C:\\Users\\YUXD\\Desktop\\open_cv\\HNY_CV\\video with depth\\calibresult.bmp";

	imshow("disparity2", dispcolor2);
	imwrite(rgbImageL_calib_result, dispcolor2);
	setMouseCallback("disparity2", onMouse, 0);

	/*Mat img1p2, img2p2;
	copyMakeBorder(rectifyImageL, img1p2, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(rectifyImageR, img2p2, 0, 0, NumDisparities, 0, IPL_BORDER_REPLICATE);
	dispf2 = disp2.colRange(NumDisparities, img2p2.cols - NumDisparities);
	dispf2.convertTo(disp9, CV_8U, 255 / (NumDisparities *16.));

	reprojectImageTo3D(dispf2, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity2", disp9);
	Mat color2(dispf2.size(), CV_8UC3);
	GenerateFalseMap(disp9, color2);//转成彩图
	imshow("disparity2", color2);*/





	cvWaitKey(1) == 27;

}








//--主函数---------------------------------------------------------------------
int main()
{
	setUseOptimized(true);
	bool simd1 = hasSIMD128();
	cout << "CPU加速器是否打开：" << simd1 << endl;
//--立体校正-------------------------------------------------------------------
	Rodrigues(rec, R);                                   //Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	int k = 1;
	double t = 0;
	while (1) {
		//--读取图片，【需要调整参数的位置4】----------------------------------------------------------------
		string rgbImageL_load = "C:\\Users\\YUXD\\Desktop\\open_cv\\HNY_CV\\video with depth\\letfframe.bmp";
		string rgbImageR_load = "C:\\Users\\YUXD\\Desktop\\open_cv\\HNY_CV\\video with depth\\rightframe.bmp";
		string rgbImageL_calib = "C:\\Users\\YUXD\\Desktop\\open_cv\\HNY_CV\\video with depth\\letfcalib.bmp";
		string rgbImageR_calib = "C:\\Users\\YUXD\\Desktop\\open_cv\\HNY_CV\\video with depth\\rightcalib.bmp";
		

		
		grayImageL = imread(rgbImageL_load, 1);
		grayImageR = imread(rgbImageR_load, 1);
		if (grayImageL.empty())continue;
		if (grayImageR.empty())continue;
		/*rgbImageL = imread(rgbImageL_load, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
		rgbImageR = imread(rgbImageR_load, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
		*/
		//namedWindow("ImageL Before Rectify", WINDOW_NORMAL);  imshow("ImageL Before Rectify", grayImageL);
		//namedWindow("ImageR Before Rectify", WINDOW_NORMAL);  imshow("ImageR Before Rectify", grayImageR);

		//--经过remap之后，左右相机的图像已经共面并且行对准----------------------------------------------
		remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
		remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
		Mat element(5, 5, CV_8U, Scalar(1));
		//Mat rectifyImageL_dilate, rectifyImageL_erode, rectifyImageL_gradient, rectifyImageL_canny;
	

		//dilate(rectifyImageL, rectifyImageL_dilate,element , Point(-1, -1));
		//erode(rectifyImageL, rectifyImageL_erode, element, Point(-1, -1));
		//rectifyImageL_gradient = rectifyImageL_dilate - rectifyImageL_erode;
		//morphologyEx(rectifyImageL, rectifyImageL_open, 1, element, Point(-1, -1)); //open化图像
		//morphologyEx(rectifyImageL, rectifyImageL_close, 2, element, Point(-1, -1));//close化图像
		//rectifyImageL = rectifyImageL_gradient;
		

		//Mat rectifyImageR_dilate, rectifyImageR_erode, rectifyImageR_gradient;
		//dilate(rectifyImageR, rectifyImageR_dilate, element, Point(-1, -1));
		//erode(rectifyImageR, rectifyImageR_erode, element, Point(-1, -1));
		//rectifyImageR_gradient = rectifyImageR_dilate - rectifyImageR_erode;
		//rectifyImageR = rectifyImageR_gradient;
		//Canny(rectifyImageL, rectifyImageL, 10, 100, 3);//边缘计算
		//Canny(rectifyImageR, rectifyImageR, 10, 100, 3);//边缘计算

		
		//--把校正结果显示出来---------------------------------------------------------------------------
		Mat rgbRectifyImageL, rgbRectifyImageR;
		//cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);
		rgbRectifyImageL = rectifyImageL;
		//cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);
		rgbRectifyImageR = rectifyImageR;

		//writeMatToFile(rectifyImageL,"rectifyImageL.txt");
		imwrite(rgbImageL_calib, rectifyImageL); 
		//imwrite("rectifyImageR.jpg", rectifyImageR);


		//namedWindow("ImageL After Rectify", WINDOW_NORMAL); imshow("ImageL After Rectify", rgbRectifyImageL);
		//namedWindow("ImageR After Rectify", WINDOW_NORMAL); imshow("ImageR After Rectify", rgbRectifyImageR);
		//namedWindow("ImageL erode", WINDOW_NORMAL); imshow("ImageL erode", rectifyImageL_erode);
		//namedWindow("ImageR erode", WINDOW_NORMAL); imshow("ImageR erode", rectifyImageR_erode);
		//namedWindow("ImageL dilate", WINDOW_NORMAL); imshow("ImageL dilate", rectifyImageL_dilate);
		//namedWindow("ImageR dilate", WINDOW_NORMAL); imshow("ImageR dilate", rectifyImageR_dilate);
		//namedWindow("ImageL gradient", WINDOW_NORMAL); imshow("ImageL gradient", rectifyImageL_gradient);
		//namedWindow("ImageR gradient", WINDOW_NORMAL); imshow("ImageR gradient", rectifyImageR_gradient);
		//namedWindow("ImageL open", WINDOW_NORMAL); imshow("ImageL open", rectifyImageL_open);
		//namedWindow("ImageL close", WINDOW_NORMAL); imshow("ImageL close", rectifyImageL_close);
		//namedWindow("ImageL canny", WINDOW_NORMAL); imshow("ImageL canny", rectifyImageL_canny);
		
		
		/*
		//--显示在同一张图上-----------------------------------------------------------------------------
		Mat canvas;
		double sf;
		int w, h;
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);
		h = cvRound(imageSize.height * sf);
		canvas.create(h, w * 2, CV_8UC3);                                             //注意通道
		
	//--左图像画到画布上-----------------------------------------------------------------------------
		Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
		resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);    //把图像缩放到跟canvasPart一样大小  
		Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                  //获得被截取的区域    
			cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
		//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                     
		//cout << "Painted ImageL" << endl;

		//--右图像画到画布上-----------------------------------------------------------------------------
		canvasPart = canvas(Rect(w, 0, w, h));                                        //获得画布的另一部分  
		resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
		Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
			cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
		//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
		//cout << "Painted ImageR" << endl;
		
		//--画上对应的线条-------------------------------------------------------------------------------
		for (int i = 0; i < canvas.rows; i += 16)
			line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
		namedWindow("rectified", WINDOW_NORMAL);  imshow("rectified", canvas);
		*/
		//--显示结果-------------------------------------------------------------------------------------
		//namedWindow("disparity", WINDOW_NORMAL);
		//namedWindow("disparity2", WINDOW_NORMAL);
		//--创建SAD窗口 Trackbar-------------------------------------------------------------------------
			//createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);

		//--创建视差唯一性百分比窗口 Trackbar------------------------------------------------------------
			//createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);

		//--创建视差窗口 Trackbar------------------------------------------------------------------------
			//createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
		double t0 = (double)getTickCount();
		//--鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)------------
		//setMouseCallback("disparity", onMouse, 0);
		//setMouseCallback("disparity2", onMouse, 0);
		//rectifyImageL = rectifyImageL_gradient;
		//rectifyImageR = rectifyImageR_gradient;
		//stereo_match_sgbm(0, 0);   //--【需要调整参数的位置5】，本行调用sgbm算法，下一行调用BM算法，二选一进行距离测量。
		//stereo_match_bm(0, 0);
		stereo_match(0, 0);
		//t = ((double)getTickCount() - t0) / getTickFrequency()+t;
		//cout <<k << "花费时间" << t << "秒" << endl;
		//cvWaitKey(10000) == 27;
	}
	waitKey(0);
	return 0;
}
