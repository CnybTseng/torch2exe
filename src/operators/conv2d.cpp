/******************************************************************
 * PyTorch to executable program (torch2exe).
 * Copyright © 2022 Zhiwei Zeng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the “Software”), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 *
 * This file is part of torch2exe.
 *
 * @file
 * @brief Convolution operator.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <vector>

#include <NvInfer.h>

#include "conv2d.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Conv2d::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Conv2d, [](void *p){((Operator*)p)->destroy();});
}

std::string Conv2d::get_name(void)
{
	return "Conv2d";
}

bool Conv2d::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
{
	const Configurer opt(cfg);
	const std::vector<std::string> inm = opt.get("input", std::vector<std::string>());
	if (1 != inm.size()) {
		LogWarn("wrong number of inputs for operator %s: %d\n", id, inm.size());
		return false;
	}
	
	std::string name;
	int index = 0;
	parse_input(inm[0], name, index);

	auto iter = ctx.output.find(name);
	if (iter == ctx.output.end()) {
		LogError("invalid input for operator %s: %s\n", id, inm[0]);
		return false;
	}

	nvinfer1::ITensor *input = iter->second[index];
	std::vector<int> wsize = opt.get("_saved_weight_sizes", std::vector<int>());
	if (4 != wsize.size()) {
		LogError("invalid weight tensor size for operator %s: %d\n", id, wsize.size());
		return false;
	}

	float *weight = ctx.weight + ctx.len_read;
	int64_t wlen = wsize[0] * wsize[1] * wsize[2] * wsize[3];
	ctx.len_read += wlen;

	float *bias = nullptr;
	int64_t blen = 0;
	std::string key = opt.get("bias", std::string(""));
	if (key != "") {
		bias = ctx.weight + ctx.len_read;
		blen = wsize[0];
		ctx.len_read += blen;
	}
	
	const int nbOutputMaps = wsize[0];
	const nvinfer1::DimsHW kernelSize {wsize[2], wsize[3]};
	const nvinfer1::Weights kernelWeights {nvinfer1::DataType::kFLOAT, weight, wlen};
	const nvinfer1::Weights biasWeights {nvinfer1::DataType::kFLOAT, bias, blen};
	
	auto conv = ctx.network->addConvolutionNd(*input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
	if (!conv) {
		LogError("addConvolutionNd for operator %s failed\n", id);
		return false;
	}

	const std::vector<int> _stride = opt.get("_saved_stride", std::vector<int>());
	if (2 != _stride.size()) {
		LogError("invalid stride size for operator %s: %d\n", id, _stride.size());
		return false;
	}

	const std::vector<int> _padding = opt.get("_saved_padding", std::vector<int>());
	if (2 != _padding.size()) {
		LogError("invalid padding size for operator %s: %d\n", id, _padding.size());
		return false;
	}
	
	const std::vector<int> _dilation = opt.get("_saved_dilation", std::vector<int>({1, 1}));
	if (2 != _dilation.size()) {
		LogError("invalid dilation size for operator %s: %d\n", id, _dilation.size());
		return false;
	}
	
	const nvinfer1::DimsHW stride {_stride[0], _stride[1]};
	const nvinfer1::DimsHW padding {_padding[0], _padding[1]};
	const nvinfer1::DimsHW dilation {_dilation[0], _dilation[1]};
	const int nbGroups = opt.get("_saved_groups", int(1));

	conv->setStrideNd(stride);
	conv->setPaddingNd(padding);
	conv->setDilationNd(dilation);
	conv->setNbGroups(nbGroups);
	conv->setName(id);
	conv->getOutput(0)->setName(id);
	ctx.output[id] = {conv->getOutput(0)};
	return true;
}

} // tensorrt
} // namespace algorithm