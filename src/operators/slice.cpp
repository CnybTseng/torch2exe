#include <vector>

#include <NvInfer.h>

#include "slice.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Slice::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Slice, [](void *p){((Operator*)p)->destroy();});
}

std::string Slice::get_name(void)
{
	return "Slice";
}

bool Slice::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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

	static const int INVALID_DIM = -100;
	const int _dim = opt.get("_saved_dim", INVALID_DIM);
	if (INVALID_DIM == _dim) {
		LogError("dim for operator %s is invalid: %d\n", id, _dim);
		return false;
	}
	
	const int _start = opt.get("_saved_start", INVALID_DIM);
	if (INVALID_DIM == _start) {
		LogError("start for operator %s is invalid: %d\n", id, _start);
		return false;
	}
	
	const int _step = opt.get("_saved_step", INVALID_DIM);
	if (INVALID_DIM == _step) {
		LogError("step for operator %s is invalid: %d\n", id, _step);
		return false;
	}

	nvinfer1::ITensor *input = iter->second[0];
	nvinfer1::DimsNCHW start {0, 0, 0, 0};
	start.d[_dim] = _start;
	
	nvinfer1::Dims size = input->getDimensions();
	size.d[_dim] /= _step;
	
	nvinfer1::DimsNCHW stride {1, 1, 1, 1};
	stride.d[_dim] = _step;
	auto slice = ctx.network->addSlice(*input, start, size, stride);
	if (!slice) {
		LogError("addSlice for operator %s failed\n", id);
		return false;
	}

	slice->setName(id);
	slice->getOutput(0)->setName(id);
	ctx.output[id] = {slice->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm