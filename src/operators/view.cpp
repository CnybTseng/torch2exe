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
 * @brief Tensor view operator.
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