#include <vector>

#include <NvInfer.h>

#include "permute.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Permute::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Permute, [](void *p){((Operator*)p)->destroy();});
}

std::string Permute::get_name(void)
{
	return "Permute";
}

bool Permute::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	auto permute = ctx.network->addShuffle(*input);
	if (!permute) {
		LogError("addShuffle for operator %s failed\n", id);
		return false;
	}
	
	const std::vector<int> _dims = opt.get("_saved_dims", std::vector<int>());
	if (_dims.size() < 1) {
		LogError("invalid dims size for operator %s: %d\n", id, _dims.size());
		return false;
	}
	
	nvinfer1::Permutation dims;
	for (size_t i = 0; i < _dims.size(); ++i) {
		dims.order[i] = _dims[i];
	}
	
	permute->setFirstTranspose(dims);
	permute->setName(id);
	permute->getOutput(0)->setName(id);
	ctx.output[id] = {permute->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm