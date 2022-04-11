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
 * @brief Object detection.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <vector>
#include <memory>

#include <cuda_runtime_api.h>

#include "algorithm.h"
#include "algorithmExt.h"
#include "nnEngine.h"
#include "operators/nnpp.h"

namespace algorithm {

class Detector : public AlgorithmExt
{
public:
	Detector();
	virtual ~Detector() = default;
protected:
	virtual bool parse_config(const char *cfg) override;
	virtual bool init(const char *cfg) override;
	virtual bool preprocess(const BlobSP &input, NNPP &nnpp, float pad_val);
	virtual bool postprocess(const NNPP &nnpp, BlobSP &output);
	virtual void destroy(void) override;
	int device;
	int batch_size;
	float nms_thresh;
	float score_thresh;
	std::vector<std::string> categories;
	tensorrt::NNEngine nne;
	std::vector<int> shape_in;
	std::vector<int> shape_out;
	std::shared_ptr<uint8_t> d_img;
	std::shared_ptr<float> d_in;
	std::shared_ptr<float> d_out;
	std::unique_ptr<float[]> h_in;
	std::unique_ptr<float[]> h_out;
private:
	virtual void reset_num_objs();
};

} // namespace algorithm

#endif // DETECTOR_H_