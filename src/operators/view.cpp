#include <vector>

#include <NvInfer.h>

#include "view.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> View::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new View, [](void *p){((Operator*)p)->destroy();});
}

std::string View::get_name(void)
{
	return "View";
}

bool View::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	auto view = ctx.network->addShuffle(*input);
	if (!view) {
		LogError("addShuffle for operator %s failed\n", id);
		return false;
	}
	
	// You must add _saved_shape for View manually!
	const std::vector<int> _shape = opt.get("_saved_shape", std::vector<int>());
	if (_shape.size() < 1) {
		LogError("invalid shape size for operator %s: %d\n", id, _shape.size());
		return false;
	}

	nvinfer1::DimsCHW dimensions;
	dimensions.nbDims = static_cast<int32_t>(_shape.size());
	for (int32_t i = 0; i < dimensions.nbDims; ++i) {
		dimensions.d[i] = _shape[i];
	}
	
	view->setReshapeDimensions(dimensions);
	view->setName(id);
	view->getOutput(0)->setName(id);
	ctx.output[id] = {view->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm