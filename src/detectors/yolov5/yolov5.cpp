#include "utils/logger.h"
#include "utils/deleter.h"
#include "yolov5.h"

namespace algorithm {
namespace detector {

std::unique_ptr<Algorithm, Deleter> YOLOV5::create(const char *cfg)
{
	LogInfo("YOLOV5::create\n");
	return std::unique_ptr<Algorithm, Deleter>(new YOLOV5, [](void *p){reinterpret_cast<YOLOV5 *>(p)->destroy();});
}

std::string YOLOV5::get_name(void)
{
	return std::string("YOLOV5");
}

bool YOLOV5::init(const char *cfg)
{
	LogInfo("YOLOV5::init\n");
	if (!Detector::init(cfg)) {
		return false;
	}

	static const float mean[3] = {0, 0, 0};
	static const float var_recip[3] = {1.f / 255, 1.f / 255, 1.f / 255};	
	if (!nnpp.set_static_param(shape_in[3], shape_in[2], true, true, true, mean, var_recip)) {
		LogError("NNPP::set_static_param failed\n");
		return false;
	}

	return true;
}

bool YOLOV5::execute(const BlobSP &input, BlobSP &output)
{
	LogDebug("YOLOV5::execute\n");
	if (!preprocess(input, nnpp, .5f)) {
		return false;
	}

	void *bindings[] = {d_in.get(), d_out.get(), d_out.get(), d_out.get(), d_out.get()};
	if (!nne.execute(bindings, batch_size, false)) {
		LogError("NNEngine execute failed\n");
		return false;
	}

	return postprocess(nnpp, output);
}

void YOLOV5::destroy(void)
{
	LogInfo("YOLOV5::destroy\n");
	Detector::destroy();
}

} // namespace detector
} // namespace algorithm