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

#include <cstdarg>
#include <initializer_list>
#include <memory>

namespace algorithm {

#pragma pack(1)

typedef char String32[32];

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
	DETECTION,	///! 物体检测输出, 对应algorithm::Object2DBoxFA
	OCR			///! 文字识别输出, 对应algorithm::LicensePlateFA
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
 * @struct Object2DBox
 * @brief 物体二维矩形框结构体.
 */
struct Object2DBox
{
	String32 category;		///! 物体类别
	Rect<uint16_t> box;		///! 物体边框
	float score;			///! 物体置信度
};

/**
 * @struct Object2DBoxFA
 * @brief 物体二维矩形框柔性数组结构体.
 */
struct Object2DBoxFA : public Blob
{
	uint16_t count;					///! 物体数量
	Object2DBox data[1];		///! 物体数据
};

/**
 * @struct LicensePlate
 * @brief 车牌结构体.
 */
struct LicensePlate
{
	String32 text;			//!< 车牌字符
	Rect<uint16_t> box;		//!< 车牌边框
	float score;			//!< 车牌置信度
};

/**
 * @struct LicensePlateFA
 * @brief 车牌柔性数组结构体.
 */
struct LicensePlateFA : public Blob
{
	uint16_t count;			//!< 车牌数量
	LicensePlate data[1];	//!< 车牌数据
};

/**
 * @struct AlgorithmNameFA
 * @brief 算法名字列表结构体.
 */
struct AlgorithmNameFA
{
	int count;			///! 算法数量
	String32 names[1];	///! 算法名字
};

#pragma pack()

using Deleter = void(*)(void *alg);
using LogCallback = void(*)(const char *format, va_list arg);
using BlobSP = std::shared_ptr<Blob>;
using ImageSP = std::shared_ptr<Image>;
using Object2DBoxFASP = std::shared_ptr<Object2DBoxFA>;
using LicensePlateFASP = std::shared_ptr<LicensePlateFA>;
using AlgorithmNameFASP = std::shared_ptr<AlgorithmNameFA>;

/**
 * @brief 向后兼容的数据类型. 老版类型将会在未来删除.
 */
using DetectorOutputData = Object2DBox;
using DetectorOutput     = Object2DBoxFA;
using DetectorOutputSP   = Object2DBoxFASP;
using AlgorithmList      = AlgorithmNameFA;
using AlgorithmListSP    = AlgorithmNameFASP;

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
 *	auto objs = std::static_pointer_cast<algorithm::Object2DBoxFA>(output);
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
	 * @brief 执行算法实例.
	 * @details 该接口用于多输入单输出应用场景.
	 * @param inputs 算法输入数据.
	 * @param output 算法输出数据.
	 * @return 算法执行状态. 成功: true, 失败: false.
	 */
	virtual bool execute(const std::initializer_list<BlobSP> &inputs, BlobSP &output) = 0;
	
	/**
	 * @brief 注册日志回调函数.
	 * @param fun 日志回调函数.
	 */
	static void register_logger(LogCallback fun);
	
	/**
	 * @brief 获取已注册的算法列表.
	 * @return 已注册的算法列表.
	 */
	static AlgorithmNameFASP get_algorithm_list(void);

	/**
	 * @brief 获取版本号. 版本号格式为: `major.minor.revision`.
	 * @param version 版本号存储空间地址.
	 * @param len 版本号存储空间字节长度. 建议存储空间大小不低于8字节.
	 * @return 版本查询状态. 成功: true, 失败: false.
	 */
	static bool get_version(char *const version, size_t len);
};

} // namespace algorithm

#endif // ALGORITHM_H_