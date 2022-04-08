#include <cstdarg>
#include <chrono>

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

struct ArgParser
{
	ArgParser() :
		min_nargs(3),
		alg_name("YOLOV5"),
		save_path("objs/"),
		loops(1)
	{}
	void parse(int argc, char *argv[])
	{
		if (argc < min_nargs) {
			fprintf(stdout, "Usage:\n\t%s cfg_path img_path [--alg_name alg_name] [--save_path save_path] [--loops loops]\n", argv[0]);
			exit(-1);
		}
		cfg_path = std::string(argv[1]);
		img_path = std::string(argv[2]);
		int i = min_nargs;
		for (; i < argc; ++i) {
			if (!strcmp(argv[i], "--alg_name")) {
				if (++i >= argc) goto FAIL;
				alg_name = std::string(argv[i]);
			} else if (!strcmp(argv[i], "--save_path")) {
				if (++i >= argc) goto FAIL;
				save_path = std::string(argv[i]);
			} else if (!strcmp(argv[i], "--loops")) {
				if (++i >= argc) goto FAIL;
				loops = atoi(argv[i]);
			} else {
				fprintf(stderr, "invalid argument name: %s\n", argv[i]);
				exit(-1);
			}
			fprintf(stdout, "optional argument: %s %s\n", argv[i - 1], argv[i]);
		}
		return;
FAIL:
		fprintf(stderr, "missing argument value: %s\n", argv[i - 1]);
		exit(-1);
	}

	const int min_nargs;
	std::string alg_name;
	std::string cfg_path;
	std::string img_path;
	std::string save_path;
	int loops;
};

int main(int argc, char *argv[])
{
	ArgParser parser;
	parser.parse(argc, argv);

	// 您可以部分控制日志系统的行为[可选操作]
	// algorithm::Algorithm::register_logger(custom_logger);
	
	// 查询已注册的算法[可选操作]
	// auto algorithms = algorithm::Algorithm::get_algorithm_list();
	// for (int i = 0; i < algorithms->count; ++i) {
	// 	fprintf(stdout, "registered: %s\n", algorithms->names[i]);
	// }

	// 创建算法实例
	auto model = algorithm::Algorithm::create(parser.alg_name.c_str());
	if (!model) {
		fprintf(stderr, "create algorithm failed\n");
		return -1;
	}
	
	// 初始化算法实例
	if (!model->init(parser.cfg_path.c_str())) {
		fprintf(stderr, "initialize algorithm failed\n");
		return -1;
	}
	
	int loops = parser.loops;
	bool save = loops == 1;
	int count = 0;
	float latency = 0;
	std::vector<cv::String> files;
	cv::glob(parser.img_path, files);
	for (size_t i = 0; i < files.size(); ++i) {
		cv::Mat mat = cv::imread(files[i]);
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
		algorithm::BlobSP output;

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
			continue;
		}
		
		// 获取输出结果
		auto objs = std::static_pointer_cast<algorithm::DetectorOutput>(output);
		
		// 在图像上渲染物体
		fprintf(stdout, "%s\n", files[i].c_str());
		render_objects(mat, objs);
		std::string filename = parser.save_path + filename_from_path(files[i]);
		cv::imwrite(filename, mat);
	}

	fprintf(stdout, "latency: %fms\n\n", latency / count); 
	return 0;
}