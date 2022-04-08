#include "utils/logger.h"
#include "utils/deleter.h"
#include "yolox.h"

namespace algorithm {
namespace detector {

std::unique_ptr<Algorithm, Deleter> YOLOX::create(const char *cfg)
{
	LogInfo("YOLOX::create\n");
	return std::unique_ptr<Algorithm, Deleter>(new YOLOX, [](void *p){reinterpret_cast<YOLOX *>(p)->destroy();});
}

std::string YOLOX::get_name(void)
{
	return std::string("YOLOX");
}

bool YOLOX::init(const char *cfg)
{	
	LogInfo("YOLOX::init\n");
	if (!Detector::init(cfg)) {
		return false;
	}

	static const float mean[3] = {0, 0, 0};
	static const float var_recip[3] = {1, 1, 1};
	if (!nnpp.set_static_param(shape_in[3], shape_in[2], false, true, false, mean, var_recip)) {
		LogError("NNPP::set_static_param failed\n");
		return false;
	}

	return true;
}

bool YOLOX::execute(const BlobSP &input, BlobSP &output)
{
	LogInfo("YOLOX::execute\n");
	if (!preprocess(input, nnpp, 114.f)) {
		return false;
	}

	void *bindings[] = {d_in.get(), d_out.get(), d_out.get(), d_out.get()};
	if (!nne.execute(bindings, batch_size, false)) {
		LogError("NNEngine execute failed\n");
		return false;
	}

	return postprocess(nnpp, output);
}

void YOLOX::destroy(void)
{
	LogInfo("YOLOX::destroy\n");
	Detector::destroy();
}

} // namespace detector
} // namespace algorithm