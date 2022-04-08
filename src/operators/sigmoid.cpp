#include <vector>

#include <NvInfer.h>

#include "sigmoid.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Sigmoid::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Sigmoid, [](void *p){((Operator*)p)->destroy();});
}

std::string Sigmoid::get_name(void)
{
	return "Sigmoid";
}

bool Sigmoid::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	static const nvinfer1::ActivationType type = nvinfer1::ActivationType::kSIGMOID;
	auto sigmoid = ctx.network->addActivation(*input, type);
	if (!sigmoid) {
		LogError("addActivation for operator %s failed\n", id);
		return false;
	}
	
	sigmoid->setName(id);
	sigmoid->getOutput(0)->setName(id);
	ctx.output[id] = {sigmoid->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm