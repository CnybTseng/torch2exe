#include <vector>

#include <NvInfer.h>

#include "upsample.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Upsample::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Upsample, [](void *p){((Operator*)p)->destroy();});
}

std::string Upsample::get_name(void)
{
	return "Upsample";
}

bool Upsample::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
{
	const Configurer opt(cfg);
	const std::vector<std::string> inm = opt.get("input", std::vector<std::string>());
	if (1 != inm.size()) {
		LogWarn("wrong number of inputs for operator %s: %d\n", id, inm.size());
		return false;
	}

	auto iter = ctx.output.find(inm[0]);
	if (iter == ctx.output.end()) {
		LogError("invalid input for operator %s: %s\n", id, inm[0]);
		return false;
	}

	nvinfer1::ITensor *input = iter->second[0];
	auto up = ctx.network->addResize(*input);
	if (!up) {
		LogError("addResize for operator %s failed\n", id);
		return false;
	}
	
	std::vector<float> _scale_factors = opt.get("_saved_scale_factors", std::vector<float>());
	if (2 != _scale_factors.size()) {
		LogError("invalid scale factors size for operator %s: %d\n", id, _scale_factors.size());
		return false;
	}
	
	static const int32_t nbScales = 4;
	const float scales[nbScales] = {1.f, 1.f, _scale_factors[0], _scale_factors[1]};
	up->setScales(scales, nbScales);
	
	static const nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
	static const bool alignCorners = true;
	
	up->setResizeMode(resizeMode);
	up->setAlignCorners(alignCorners);
	up->setName(id);
	up->getOutput(0)->setName(id);
	ctx.output[id] = {up->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm