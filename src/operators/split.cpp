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
 * @brief Tensor split operator.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date May 7, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <vector>

#include <NvInfer.h>

#include "split.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Split::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Split, [](void *p){((Operator*)p)->destroy();});
}

std::string Split::get_name(void)
{
	return "Split";
}

bool Split::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	const int _dim = opt.get("_saved_dim", int(-5));
	if (-5 == _dim) {
		LogError("get dim for operator %s failed\n", id);
		return false;
	}
	
	const int _split_size = opt.get("_saved_split_size", int(0));
	if (_split_size <= 0) {
		LogError("invalid split size for operator %s: %d\n", id, _split_size);
		return false;
	}
	
	nvinfer1::DimsNCHW start {0, 0, 0, 0};
	nvinfer1::Dims size = input->getDimensions();
	const int chunks = size.d[_dim] / _split_size;
	size.d[_dim] = _split_size;
	const nvinfer1::DimsNCHW stride {1, 1, 1, 1};
	
	std::vector<nvinfer1::ITensor *> outputs;
	for (int i = 0; i < chunks; ++i) {
		auto slice = ctx.network->addSlice(*input, start, size, stride);
		if (!slice) {
			LogError("addSlice for operator %s failed\n", id);
			return false;
		}
		start.d[_dim] += _split_size;
		const std::string name = std::string(id) + "." + std::to_string(i);
		slice->setName(name.c_str());
		slice->getOutput(0)->setName(name.c_str());
		outputs.push_back(slice->getOutput(0));
	}
	
	ctx.output[id] = outputs;
	return true;
}

} // namespace tensorrt
} // namespace algorithm