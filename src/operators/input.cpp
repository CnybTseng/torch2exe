#include <vector>

#include <NvInfer.h>

#include "input.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Input::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Input, [](void *p){((Operator*)p)->destroy();});
}

std::string Input::get_name(void)
{
	return "Input";
}

bool Input::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
{
	const Configurer opt(cfg);
	const std::vector<int> shape = opt.get("shape", std::vector<int>());
	if (4 != shape.size()) {
		LogError("input shape for operator %s is invalid: %d\n", id, shape.size());
		return false;
	}

	const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
	const nvinfer1::Dims4 dimensions {shape[0], shape[1], shape[2], shape[3]};
	auto tensor = ctx.network->addInput(id, type, dimensions);
	if (!tensor) {
		LogError("addInput for operator %s failed\n", id);
		return false;
	}
	
	tensor->setName(id);
	ctx.output[id] = {tensor};
	return true;
}

} // namespace tensorrt
} // namespace algorithm