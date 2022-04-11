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
 * @brief YOLOV5.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include "utils/logger.h"
#include "utils/deleter.h"
#include "yolov5.h"

namespace algorithm {
namespace detector {

std::unique_ptr<Algorithm, Deleter> YOLOV5::create(const char *cfg)
{
	LogInfo("YOLOV5::create\n");
	return std::unique_ptr<Algorithm, Deleter>(new YOLOV5, [](void *p){reinterpret_cast<YOLOV5 *>(p)->destroy();});
}

std::string YOLOV5::get_name(void)
{
	return std::string("YOLOV5");
}

bool YOLOV5::init(const char *cfg)
{
	LogInfo("YOLOV5::init\n");
	if (!Detector::init(cfg)) {
		return false;
	}

	static const float mean[3] = {0, 0, 0};
	static const float var_recip[3] = {1.f / 255, 1.f / 255, 1.f / 255};	
	if (!nnpp.set_static_param(shape_in[3], shape_in[2], true, true, true, mean, var_recip)) {
		LogError("NNPP::set_static_param failed\n");
		return false;
	}

	return true;
}

bool YOLOV5::execute(const BlobSP &input, BlobSP &output)
{
	LogDebug("YOLOV5::execute\n");
	if (!preprocess(input, nnpp, .5f)) {
		return false;
	}

	void *bindings[] = {d_in.get(), d_out.get(), d_out.get(), d_out.get(), d_out.get()};
	if (!nne.execute(bindings, batch_size, false)) {
		LogError("NNEngine execute failed\n");
		return false;
	}

	return postprocess(nnpp, output);
}

void YOLOV5::destroy(void)
{
	LogInfo("YOLOV5::destroy\n");
	Detector::destroy();
}

} // namespace detector
} // namespace algorithm