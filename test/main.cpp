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
#include <chrono>
#include <thread>

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

int main(int argc, char *argv[])
{
	const char *keys =
		"{ h             help   |                              | print help message }"
        "{ alg_name             | YOLOX                        | the algorithm need to be created }"
        "{ cfg_path             | ../config/YOLOX/yolox_m.json | algorithm configuration file path }"
        "{ img_path             | ../../imgs/station.jpeg      | image path }"
        "{ loops                | 1                            | running loops }"
        "{ save_path            | objs/                        | output image storage path }"
		"{ fps                  |                              | fixed fps}";
	
	cv::CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help") || !cmd.check()) {
        cmd.printMessage();
        cmd.printErrors();
        return 0;
    }

	// 您可以部分控制日志系统的行为[可选操作]
	// algorithm::Algorithm::register_logger(custom_logger);
	
	// 查询已注册的算法[可选操作]
	// auto algorithms = algorithm::Algorithm::get_algorithm_list();
	// for (int i = 0; i < algorithms->count; ++i) {
	// 	fprintf(stdout, "registered: %s\n", algorithms->names[i]);
	// }

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
	
	int loops = cmd.get<int>("loops");
	bool save = loops == 1;
	int count = 0;
	float latency = 0;
	std::vector<cv::String> files;
	cv::glob(cmd.get<std::string>("img_path"), files);
	std::string save_path = cmd.get<std::string>("save_path");
	float fixed_latency = -1;
	if (cmd.has("fps")) {
		int fps = cmd.get<int>("fps");
		fixed_latency = 1000.f / fps;
	}
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
		if (!model->execute(image, output)) break;
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
		std::string filename = save_path + filename_from_path(files[i]);
		cv::imwrite(filename, mat);
	}

	fprintf(stdout, "latency: %fms\n\n", latency / count); 
	return 0;
}