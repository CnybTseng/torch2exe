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
 * @brief Tensor transposition operator.
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

#include "transpose.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Transpose::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Transpose, [](void *p){((Operator*)p)->destroy();});
}

std::string Transpose::get_name(void)
{
	return "Transpose";
}

bool Transpose::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
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
	auto permute = ctx.network->addShuffle(*input);
	if (!permute) {
		LogError("addShuffle for operator %s failed\n", id);
		return false;
	}
	
	const int _dim0 = opt.get("_saved_dim0", int(-5));
	if (-5 == _dim0) {
		LogError("get dim0 for operator %s failed\n", id);
		return false;
	}
	
	const int _dim1 = opt.get("_saved_dim1", int(-5));
	if (-5 == _dim1) {
		LogError("get dim1 for operator %s failed\n", id);
		return false;
	}

	nvinfer1::Permutation dims;
	for (int i = 0; i < nvinfer1::Dims::MAX_DIMS; ++i) {
		dims.order[i] = i;
	}

	dims.order[_dim0] = _dim1;
	dims.order[_dim1] = _dim0;
	
	permute->setFirstTranspose(dims);
	permute->setName(id);
	permute->getOutput(0)->setName(id);
	ctx.output[id] = {permute->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm