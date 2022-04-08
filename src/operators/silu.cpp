#include <vector>

#include <NvInfer.h>

#include "silu.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> SiLU::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new SiLU, [](void *p){((Operator*)p)->destroy();});
}

std::string SiLU::get_name(void)
{
	return "SiLU";
}

bool SiLU::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	const nvinfer1::ActivationType type = nvinfer1::ActivationType::kSIGMOID;
	
	auto sigmoid = ctx.network->addActivation(*input, type);
	if (!sigmoid) {
		LogError("addActivation for operator %s failed\n", id);
		return false;
	}
	
	const std::string sigmoid_name = std::string(id) + ".0";
	sigmoid->setName(sigmoid_name.c_str());
	sigmoid->getOutput(0)->setName(sigmoid_name.c_str());
	
	const nvinfer1::ElementWiseOperation op = nvinfer1::ElementWiseOperation::kPROD;
	auto silu = ctx.network->addElementWise(*input, *sigmoid->getOutput(0), op);
	if (!silu) {
		LogError("addElementWise for operator %s failed\n", id);
		return false;
	}

	const std::string prod_name = std::string(id) + ".1";
	silu->setName(prod_name.c_str());
	silu->getOutput(0)->setName(prod_name.c_str());
	ctx.output[id] = {silu->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm