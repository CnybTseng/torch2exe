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
 * @brief Tensor addition operator.
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

#include "add.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> Add::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new Add, [](void *p){((Operator*)p)->destroy();});
}

std::string Add::get_name(void)
{
	return "Add";
}

bool Add::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
{
	const Configurer opt(cfg);
	const std::vector<std::string> inm = opt.get("input", std::vector<std::string>());
	if (inm.size() < 1) {
		LogWarn("wrong number of inputs for operator %s: %d\n", id, inm.size());
		return false;
	}

	std::vector<nvinfer1::ITensor *> inputs;
	const int32_t nbInputs = static_cast<int32_t>(inm.size());
	for (auto in : inm) {
		auto iter = ctx.output.find(in);
		if (iter == ctx.output.end()) {
			LogError("invalid input for operator %s: %s\n", id, inm[0]);
			return false;
		}
		inputs.emplace_back(iter->second[0]);
	}

	nvinfer1::ITensor *total = inputs[0];
	const nvinfer1::ElementWiseOperation op = nvinfer1::ElementWiseOperation::kSUM;
	for (size_t i = 1; i < inputs.size(); ++i) {
		auto sum = ctx.network->addElementWise(*total, *inputs[i], op);
		if (!sum) {
			LogError("addElementWise for operator %s failed\n", id);
			return false;
		}
		const std::string sum_name = std::string(id) + "." + std::to_string(i - 1);
		sum->setName(sum_name.c_str());
		sum->getOutput(0)->setName(sum_name.c_str());
		total = sum->getOutput(0);
	}

	ctx.output[id] = {total};
	return true;
}

} // namespace tensorrt
} // namespace algorithm