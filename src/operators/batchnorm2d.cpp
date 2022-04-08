#include <cmath>
#include <vector>

#include <NvInfer.h>

#include "batchnorm2d.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> BatchNorm2d::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new BatchNorm2d, [](void *p){((Operator*)p)->destroy();});
}

std::string BatchNorm2d::get_name(void)
{
	return "BatchNorm2d";
}

bool BatchNorm2d::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	
	const float eps = opt.get("_saved_epsilon", .001f);
	
	nvinfer1::ITensor *input = iter->second[0];
	const int64_t count = input->getDimensions().d[1];

	float *weight = ctx.weight + ctx.len_read;
	ctx.len_read += count;
	
	float *bias = ctx.weight + ctx.len_read;
	ctx.len_read += count;
	
	float *runnint_mean = ctx.weight + ctx.len_read;
	ctx.len_read += count;
	
	float *running_var = ctx.weight + ctx.len_read;
	ctx.len_read += count;
	
	std::shared_ptr<float[]> _shift(new (std::nothrow) float[count]);
	if (!_shift) {
		LogError("allocate shift buffer failed\n");
		return false;
	}
	
	std::shared_ptr<float[]> _scale(new (std::nothrow) float[count]);
	if (!_scale) {
		LogError("allocate scale buffer failed\n");
		return false;
	}
	
	std::shared_ptr<float[]> _power(new (std::nothrow) float[count]);
	if (!_power) {
		LogError("allocate scale buffer failed\n");
		return false;
	}
	
	for (int64_t i = 0; i < count; ++i) {
		_scale.get()[i] = weight[i] / sqrt(running_var[i] + eps);
		_shift.get()[i] = bias[i] - weight[i] * runnint_mean[i] / sqrt(running_var[i] + eps);
		_power.get()[i] = 1;
	}

	const nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
	const nvinfer1::Weights shift {nvinfer1::DataType::kFLOAT, _shift.get(), count};
	const nvinfer1::Weights scale {nvinfer1::DataType::kFLOAT, _scale.get(), count};
	const nvinfer1::Weights power {nvinfer1::DataType::kFLOAT, _power.get(), count};

	auto bn = ctx.network->addScale(*input, mode, shift, scale, power);
	if (!bn) {
		LogError("addScale for operator %s failed\n", id);
		return false;
	}
	
	// keep these parameters alive before engine generated
	ctx.derived_weights.emplace_back(_scale);
	ctx.derived_weights.emplace_back(_shift);
	ctx.derived_weights.emplace_back(_power);
	
	bn->setName(id);
	bn->getOutput(0)->setName(id);
	ctx.output[id] = {bn->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm