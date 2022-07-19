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
 * @brief torch2exe demonstration module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cstdarg>
#include <cstring>
#include <chrono>
#include <thread>

#include <opencv2/freetype.hpp>
#include <opencv2/opencv.hpp>

#include "algorithm.h"

static void custom_logger(const char *format, va_list arg)
{
	FILE *fp = fopen("algorithm.log", "a");
	if (fp) {
		vfprintf(fp, format, arg);
		fclose(fp);
	}
}

static std::string filename_from_path(const std::string path)
{
	char sep = '/';
#ifdef _WIN32
	sep = '\\';
#endif
	size_t i = path.rfind(sep, path.length());
	if (i != std::string::npos) {
		return path.substr(i + 1, path.length() - i);
	}
	return "";
}

static void render_objects(cv::Mat &img, const algorithm::DetectorOutputSP &objs)
{
	if (!objs) {
		fprintf(stdout, "no object\n");
		return;
	}
	for (int j = 0; j < objs->count; ++j) {
		fprintf(stdout, "category:%16s box:[%4u %4u %4u %4u] score:%.2f\n",
			objs->data[j].category,
			objs->data[j].box.x,
			objs->data[j].box.y,
			objs->data[j].box.width,
			objs->data[j].box.height,
			objs->data[j].score
		);
		
		cv::Rect rect(objs->data[j].box.x, objs->data[j].box.y, objs->data[j].box.width, objs->data[j].box.height);
		cv::rectangle(img, rect, cv::Scalar(0xFF, 0x00, 0x00), 1);
		
		char text[32];
        snprintf(text, sizeof(text), "%s:%.0f", objs->data[j].category, objs->data[j].score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = objs->data[j].box.x;
        int y = objs->data[j].box.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > img.cols) x = img.cols - label_size.width;

        cv::rectangle(img,
			cv::Rect(x, y, label_size.width, label_size.height + baseLine),
			cv::Scalar(0x00, 0x00, 0x00), -1);

        cv::putText(img, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0xFF, 0xFF, 0xFF));
	}
}

static void render_lps(cv::Mat &img, const algorithm::LicensePlateFASP &lps)
{
	if (!lps) {
		fprintf(stdout, "no license plate\n");
		return;
	}

	cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
	ft2->loadFontData("fonts/simhei.ttf", 0);
	const int fontHeight = 20;
	const int thickness = -1;
	const int linestyle = 8;
	for (int i = 0; i < lps->count; ++i) {
		cv::Rect rect(lps->data[i].box.x, lps->data[i].box.y, lps->data[i].box.width, lps->data[i].box.height);
		cv::rectangle(img, rect, cv::Scalar(0x00, 0xFF, 0xFF), 2);
		int x = lps->data[i].box.x + lps->data[i].box.width;
		int y = lps->data[i].box.y + lps->data[i].box.height;
		ft2->putText(img, lps->data[i].text, cv::Point(x, y),
			fontHeight, cv::Scalar(0x00, 0xFF, 0xFF), thickness, linestyle, true);
		fprintf(stdout, "lp %d: %u %u %u %u %f %s\n", i, lps->data[i].box.x, lps->data[i].box.y,
			lps->data[i].box.width, lps->data[i].box.height, lps->data[i].score, lps->data[i].text);
	}
}

int main(int argc, char *argv[])
{
	const char *keys =
		"{ h             help   |                              | print help message }"
        "{ alg_name             | YOLOX                        | the algorithm need to be created }"
        "{ cfg_path             | ../config/YOLOX/yolox_m.json | algorithm configuration file path }"
        "{ img_path             | ../../imgs/station.jpeg      | image path }"
        "{ loops                | 1                            | running loops }"
        "{ save_path            | objs/                        | output image storage path }"
		"{ fps                  |                              | fixed fps}"
		"{ with_lprec           | 0                            | with license plate recognition }"
		"{ init_lpdet           | 1                            | init license plate detection result }";
	
	cv::CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help") || !cmd.check()) {
        cmd.printMessage();
        cmd.printErrors();
        return 0;
    }

	// 您可以部分控制日志系统的行为[可选操作]
	// algorithm::Algorithm::register_logger(custom_logger);
	
	// 查询已注册的算法[可选操作]
	auto algorithms = algorithm::Algorithm::get_algorithm_list();
	for (int i = 0; i < algorithms->count; ++i) {
		fprintf(stdout, "registered algorithm: %s\n", algorithms->names[i]);
	}
	
	// 查询算法基础模块版本号
	char version[8] = {0};
	if (!algorithm::Algorithm::get_version(version, sizeof(version))) {
		fprintf(stderr, "get algorithm version failed\n");
		return -1;
	}
	fprintf(stdout, "algorithm version: %s\n", version);

	// 创建算法实例
	auto model = algorithm::Algorithm::create(cmd.get<std::string>("alg_name").c_str());
	if (!model) {
		fprintf(stderr, "create algorithm failed\n");
		return -1;
	}
	
	// 初始化算法实例
	if (!model->init(cmd.get<std::string>("cfg_path").c_str())) {
		fprintf(stderr, "initialize algorithm failed\n");
		return -1;
	}
	
	// 创建车牌识别算法实例
	auto lp_recognizer = algorithm::Algorithm::create("PPOCR");
	if (!lp_recognizer) {
		fprintf(stderr, "create license plate recognizer failed\n");
		return -1;
	}
	
	// 初始化车牌识别算法实例
	if (!lp_recognizer->init(cmd.get<std::string>("cfg_path").c_str())) {
		fprintf(stderr, "initialize license plate recognizer failed\n");
		return -1;
	}
	
	int loops = cmd.get<int>("loops");
	bool save = loops == 1;	//!< 非延迟测试模式, 保存结果
	int count = 0;
	float latency = 0;
	std::vector<cv::String> files;
	cv::glob(cmd.get<std::string>("img_path"), files);
	if (files.size() == 0) {
		fprintf(stderr, "no image\n");
		return -1;
	}
	std::string save_path = cmd.get<std::string>("save_path");
	float fixed_latency = -1;
	if (cmd.has("fps")) {
		int fps = cmd.get<int>("fps");
		fixed_latency = 1000.f / fps;
	}
	bool with_lprec = cmd.get<int>("with_lprec") != 0;
	bool init_lpdet = cmd.get<int>("init_lpdet") != 0;
	cv::Mat mat = cv::imread(files[0]);
	for (size_t i = 0; i < files.size(); ++i) {
		if (save) {
			mat = cv::imread(files[i]);
		}
		if (mat.empty()) {
			fprintf(stderr, "read image %s failed\n", files[i].c_str());
			return -1;
		}
		
		// 创建输入图像
		// 如果输入图像的尺寸永远不变, 应在进入该循环体之前创建图像
		const size_t size = mat.rows * mat.step;
		auto image = algorithm::Image::alloc(mat.rows, mat.cols, static_cast<uint16_t>(mat.step), true);
		if (!image) {
			fprintf(stderr, "allocate image buffer failed\n");
			return -1;
		}

		// 更新图像数据
		memcpy(image->data, mat.data, size);		
		algorithm::BlobSP output(nullptr);

		// 执行算法实例
		auto start = std::chrono::high_resolution_clock::now();
		if ((!with_lprec) || (with_lprec && init_lpdet)) {
			if (!model->execute(image, output)) {
				break;
			}
		} // else: 只测试车牌识别
		algorithm::BlobSP lps(nullptr);
		if (with_lprec) {
			// 执行车牌识别算法实例
			algorithm::BlobSP dets = init_lpdet ? output : nullptr;
			const std::initializer_list<algorithm::BlobSP> inputs = {image, dets};
			if (!lp_recognizer->execute(inputs, lps)) {
				break;
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		float duration = std::chrono::duration<float, std::milli>(end - start).count();
		latency += duration;
		++count;

		// 用于测试算法延迟
		if (!save) {
			if (i == files.size() - 1 && --loops) {
				i = -1;
			}
			if (fixed_latency > 0 && duration < fixed_latency) {
				int sleeps = static_cast<int>(fixed_latency - duration);
				std::this_thread::sleep_for(std::chrono::milliseconds(sleeps));
			}
			continue;
		}
		
		// 获取输出结果
		auto objs = std::static_pointer_cast<algorithm::DetectorOutput>(output);
		
		// 在图像上渲染物体
		fprintf(stdout, "%s\n", files[i].c_str());
		render_objects(mat, objs);
		if (with_lprec) {
			auto lps_ = std::static_pointer_cast<algorithm::LicensePlateFA>(lps);
			// 渲染和打印车牌识别结果
			render_lps(mat, lps_);
		}
		std::string filename = save_path + filename_from_path(files[i]);
		cv::imwrite(filename, mat);
	}

	fprintf(stdout, "latency: %fms\n\n", latency / count); 
	return 0;
}