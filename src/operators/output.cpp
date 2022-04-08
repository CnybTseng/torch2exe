#include <vector>

#include <NvInfer.h>

#include "output.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Output::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Output, [](void *p){((Operator*)p)->destroy();});
}

std::string Output::get_name(void)
{
	return "Output";
}

bool Output::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	nvinfer1::Dims dims = input->getDimensions();
	std::string strs = "";
	for (int32_t i = 0; i < dims.nbDims; ++i) {
		strs = strs + std::to_string(dims.d[i]) + " ";
	}
	LogDebug("%s output size: %s\n", input->getName(), strs.c_str());
	ctx.network->markOutput(*input);
	return true;
}

} // namespace tensorrt
} // namespace algorithm