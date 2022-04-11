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
 * @brief Neural network preprocessing module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifndef NNPP_H_
#define NNPP_H_

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "operators/plugins/common.h"

namespace algorithm {

class NNPP
{
public:
	NNPP();
	virtual bool set_input_size(int inw, int inh);
	virtual bool set_static_param(int outw, int outh, bool align_center, bool kar, bool reverse_channel,
		const float *mean, const float *var_recip);
	virtual void forward(const uint8_t *d_in, float *d_out, cudaStream_t stream);
	virtual void backward_inplace(std::vector<Detection> &dets, const std::vector<int> &mask=std::vector<int>()) const;
	virtual ~NNPP();
private:
	void set_param();
	int inw;
	int inh;
	int outw;
	int outh;
	bool align_center;
	bool kar;
	bool reverse_channel;
	int padw;
	int padh;
	int w;
	int h;
	float scale;
	std::unique_ptr<float[]> h_mean;
	std::unique_ptr<float[]> h_var_recip;
	std::shared_ptr<float> d_mean;
	std::shared_ptr<float> d_var_recip;
};

} // namespace algorithm

#endif // NNPP_H_