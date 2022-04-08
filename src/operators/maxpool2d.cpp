#include <vector>

#include <NvInfer.h>

#include "maxpool2d.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> MaxPool2d::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new MaxPool2d, [](void *p){((Operator*)p)->destroy();});
}

std::string MaxPool2d::get_name(void)
{
	return "MaxPool2d";
}

bool MaxPool2d::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	const nvinfer1::PoolingType type = nvinfer1::PoolingType::kMAX;
	
	std::vector<int> _kernel_size = opt.get("_saved_kernel_size", std::vector<int>());
	if (2 != _kernel_size.size()) {
		LogError("invalid kernel size for operator %s: %d\n", id, _kernel_size.size());
		return false;
	}
	
	const nvinfer1::DimsHW windowSize {_kernel_size[0], _kernel_size[1]};
	auto pool = ctx.network->addPoolingNd(*input, type, windowSize);
	if (!pool) {
		LogError("addPoolingNd for operator %s failed\n", id);
		return false;
	}
	
	std::vector<int> _stride = opt.get("_saved_stride", std::vector<int>());
	if (2 != _stride.size()) {
		LogError("invalid stride size for operator %s: %d\n", id, _stride.size());
		return false;
	}
	
	std::vector<int> _padding = opt.get("_saved_padding", std::vector<int>());
	if (2 != _padding.size()) {
		LogError("invalid padding size for operator %s: %d\n", id, _padding.size());
		return false;
	}
	
	const nvinfer1::DimsHW stride {_stride[0], _stride[1]};
	const nvinfer1::DimsHW padding {_padding[0], _padding[1]};
	
	pool->setStrideNd(stride);
	pool->setPaddingNd(padding);
	pool->setName(id);
	pool->getOutput(0)->setName(id);
	ctx.output[id] = {pool->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm