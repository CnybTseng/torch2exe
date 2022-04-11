/******************************************************************
 * PyTorch to executable program (torch2exe).
 * Copyright © 2022 Zhiwei Zeng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the “Software”), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 *
 * This file is part of torch2exe.
 *
 * @file
 * @brief Algorithm interface module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifndef ALGORITHM_H_
#define ALGORITHM_H_

#if defined _WIN32 || defined __CYGWIN__
  #define ALGORITHM_HELPER_DLL_IMPORT __declspec(dllimport)
  #define ALGORITHM_HELPER_DLL_EXPORT __declspec(dllexport)
  #define ALGORITHM_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define ALGORITHM_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define ALGORITHM_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define ALGORITHM_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define ALGORITHM_HELPER_DLL_IMPORT
    #define ALGORITHM_HELPER_DLL_EXPORT
    #define ALGORITHM_HELPER_DLL_LOCAL
  #endif
#endif

#ifdef ALGORITHM_DLL
  #ifdef ALGORITHM_DLL_EXPORTS
    #define ALGORITHM_API ALGORITHM_HELPER_DLL_EXPORT
  #else
    #define ALGORITHM_API ALGORITHM_HELPER_DLL_IMPORT
  #endif // ALGORITHM_DLL_EXPORTS
  #define ALGORITHM_LOCAL ALGORITHM_HELPER_DLL_LOCAL
#else
  #define ALGORITHM_API
  #define ALGORITHM_LOCAL
#endif // ALGORITHM_DLL

#define NAME_MAXLEN 32

#include <cstdarg>
#include <memory>

namespace algorithm {

#pragma pack(1)

typedef char Name[NAME_MAXLEN];

/**
 * @struct Point2D
 * @brief 二维点结构体.
 */
template <typename T>
struct Point2D
{
	T x;	///! 横坐标
	T y;	///! 纵坐标
};

/**
 * @struct Rect
 * @brief 矩形框结构体.
 */
template <typename T>
struct Rect
{
	T x;		///! 左上角横坐标
	T y;		///! 左上角纵坐标
	T width;	///! 宽度
	T height;	///! 高度
};

/**
 * @enum BlobType
 * @brief 数据块类型.
 */
enum BlobType
{
	IMAGE,		///! 图像, 对应algorithm::Image
	DETECTION	///! 物体检测输出, 对应algorithm::DetectorOutput
};

/**
 * @struct Blob
 * @brief 算法模块输入输出数据结构体.
 * @warning 所有具体的输入输出结构体均继承本结构体.
 */
struct Blob
{
	char type;	///! 数据块类型, 参考algorithm::BlobType
};

/**
 * @struct Image
 * @brief BGR24图像结构体.
 */
struct ALGORITHM_API Image : public Blob
{
	/**
	 * @brief 分配Image结构体内存.
	 * @param h 图像的高度.
	 * @param w 图像的宽度.
	 * @param s 图像的行字节步长. 对于BGR24图像而言, s=(w + pad) * 3 * sizeof(char),
	 *  pad是图像每行末尾像素单位的填充量. 参数s对应OpenCV中Mat的step参数.
	 * @param pinned 分配page-locked内存(true)还是一般内存(false).
	 *  pinned=true仅适用于CUDA加速的应用场景.
	 * @return 智能指针形式的Image.
	 */
	static std::shared_ptr<Image> alloc(uint16_t h, uint16_t w, uint16_t s, bool pinned=false);

	uint16_t width;		///! 图像宽度
	uint16_t height;	///! 图像高度
	uint16_t stride;	///! 图像扫描行字节步长
	char data[1];		///! 图像数据数组
};

/**
 * @struct DetectorOutputData
 * @brief 物体检测器输出数据结构体.
 */
struct DetectorOutputData
{
	Name category;			///! 物体类别
	Rect<uint16_t> box;		///! 物体边框
	float score;			///! 物体置信度
};

/**
 * @struct DetectorOutput
 * @brief 物体检测器输出结构体.
 */
struct DetectorOutput : public Blob
{
	uint16_t count;					///! 物体数量
	DetectorOutputData data[1];		///! 物体数据
};

/**
 * @struct LLA
 * @brief 地理坐标结构体.
 */
struct LLA
{
	float latitude;		///! 纬度[`°`]
	float longitude;	///! 经度[`°`]
	float altitude;		///! 海拔[`m`]
};

/**
 * @struct LLA2CameraInput
 * @brief LLA2Camera算法模块输入结构体.
 */
struct LLA2CameraInput : public Blob
{
	struct
	{
		LLA lla;			///! 无人机地理坐标
		float roll;			///! 无人机在NED坐标系下的滚转角[`°`]
		float pitch;		///! 无人机在NED坐标系下的俯仰角[`°`]
		float yaw;			///! 无人机在NED坐标系下的偏航角[`°`]
	} uav;
	struct
	{
		float tilt;			///! 摄像机在机体坐标系下的俯仰角[`°`]
		float pan;			///! 摄像机在机体坐标系下的偏航角[`°`]
		float roll;			///! 摄像机在机体坐标系下的滚转角[`°`]
		uint16_t hreso;		///! 图像水平分辨率
		uint16_t vreso;		///! 图像垂直分辨率
		float fx;			///! 焦距
		float fy;
		float cx;			///! 主点坐标
		float cy;
		float k1;			///! 镜头径向畸变系数
		float k2;
		float k3;
		float k4;
		float k5;
		float k6;
		float p1;			///! 镜头切向畸变系数
		float p2;
		float s1;			///! 镜头棱镜畸变系数
		float s2;
		float s3;
		float s4;
		LLA lla[4];			///! 图像四角地理坐标, 存储顺序: 左上角, 右上角, 右下角, 左下角
	} camera;
	uint16_t count;			///! 目标个数
	LLA target[1];			///! 目标地理坐标
};

/**
 * @struct LLA2CameraOutput
 * @brief LLA2Camera算法模块输出结构体.
 */
struct LLA2CameraOutput : public Blob
{
	uint16_t count;				///! 目标个数
	Point2D<uint16_t> data[1];	///! 目标像素坐标
};

/**
 * @struct AlgorithmList
 * @brief 算法名字列表结构体.
 */
struct AlgorithmList
{
	int count;		///! 算法数量
	Name names[1];	///! 算法名字
};

#pragma pack()

using Deleter = void(*)(void *alg);
using LogCallback = void(*)(const char *format, va_list arg);
using BlobSP = std::shared_ptr<Blob>;
using ImageSP = std::shared_ptr<Image>;
using DetectorOutputSP = std::shared_ptr<DetectorOutput>;
using AlgorithmListSP = std::shared_ptr<AlgorithmList>;

/**
 * @class Algorithm
 * @brief 普通用户使用的算法接口类.
 * @details 普通用户使用的算法接口类. 框架内算法必须实现init和execute方法.
 *  调用示例:
 *  ...
 *  auto detector = algorithm::Algorithm::create(...);
 *  auto image = algorithm::Image::alloc(...);
 *  algorithm::BlobSP output;
 *	detector->execute(image, output);
 *	auto objs = std::static_pointer_cast<algorithm::DetectorOutput>(output);
 *  ...
 */
class ALGORITHM_API Algorithm
{
public:
	Algorithm() = default;
	virtual ~Algorithm() = default;
	
	/**
	 * @brief 创建算法实例.
	 * @details 根据算法名字创建算法实例. 算法名字必须和算法注册的名字一致.
	 *  系统支持的算法和算法的配置项参考技术文档: `算法白皮书`. 也可以通过接口函数
	 *  Algorithm::get_algorithm_list()查询系统支持的算法.
	 *
	 * @return 智能指针形式的算法实例.
	 */
	static std::unique_ptr<Algorithm, Deleter> create(const char *name);

	/**
	 * @brief 初始化算法实例.
	 * @details 初始化算法实例. 配置为JSON格式字符串.
	 *  比如:
	 *  {
     *     name: "YOLOv3",
	 *     num_class: 80,
	 *     anchors: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
	 *     ...
	 *  }
	 *
	 * @param cfg 算法配置, 为JSON格式字符串.
	 * @return 算法初始化状态. 成功: true, 失败: false.
	 */
	virtual bool init(const char *cfg) = 0;

	/**
	 * @brief 执行算法实例.
	 * @param input 算法输入数据.
	 * @param output 算法输出数据.
	 * @return 算法执行状态. 成功: true, 失败: false.
	 */
	virtual bool execute(const BlobSP &input, BlobSP &output) = 0;
	
	/**
	 * @brief 注册日志回调函数.
	 * @param fun 日志回调函数.
	 */
	static void register_logger(LogCallback fun);
	
	/**
	 * @brief 获取已注册的算法列表.
	 * @return 已注册的算法列表.
	 */
	static AlgorithmListSP get_algorithm_list(void);
};

} // namespace algorithm

#endif // ALGORITHM_H_