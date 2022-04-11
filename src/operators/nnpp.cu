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

#include <algorithm>
#include <cstdio>

#include "nnpp.h"
#include "utils/cuda.h"
#include "utils/logger.h"

namespace algorithm {

__global__ void bilinear_interpolate(const uchar3 *in, float *out,
	int inw, int inh, int outw, int outh, int w, int h, int padw, int padh,
	float scale, float *mean, float *var_recip, bool reverse_channel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > w * h)
		return;
	
	int y = idx / w;
	int x = idx - y * w;

	int x_out = x + padw;
	if (x_out < 0 || x_out > outw - 1)
		return;
	int y_out = y + padh;
	if (y_out < 0 || y_out > outh - 1)
		return;
	int out_stride = outw * outh;

	float x_inf = scale * (x + .5f) - .5f;
	float y_inf = scale * (y + .5f) - .5f;
	int x_in = int(x_inf);
	if (x_in < 0 || x_in > inw - 2)
		return;
	int y_in = int(y_inf);
	if (y_in < 0 || y_in > inh - 2)
		return;
	
	float dx = x_inf - x_in;
	float dy = y_inf - y_in;
	
	float w4 = dx * dy;
	float w1 = 1 - dx - dy + w4;
	float w2 = dx - w4;
	float w3 = dy - w4;
	
	const uchar3 vin1 = in[y_in * inw + x_in];
	const uchar3 vin2 = in[y_in * inw + x_in + 1];
	const uchar3 vin3 = in[(y_in + 1) * inw + x_in];
	const uchar3 vin4 = in[(y_in + 1) * inw + x_in + 1];
	
	float vout1 = w1 * vin1.x + w2 * vin2.x + w3 * vin3.x + w4 * vin4.x;
	float vout2 = w1 * vin1.y + w2 * vin2.y + w3 * vin3.y + w4 * vin4.y;
	float vout3 = w1 * vin1.z + w2 * vin2.z + w3 * vin3.z + w4 * vin4.z;
	
	if (reverse_channel) {
		out[y_out * outw + x_out + (out_stride << 1)] = (vout1 - mean[0]) * var_recip[0];
		out[y_out * outw + x_out + out_stride] = (vout2 - mean[1]) * var_recip[1];
		out[y_out * outw + x_out] = (vout3 - mean[2]) * var_recip[2];
	} else {
		out[y_out * outw + x_out] = (vout1 - mean[0]) * var_recip[0];
		out[y_out * outw + x_out + out_stride] = (vout2 - mean[1]) * var_recip[1];
		out[y_out * outw + x_out + (out_stride << 1)] = (vout3 - mean[2]) * var_recip[2];
	}
}

NNPP::NNPP() :
	padw(0),
	padh(0)
{
}

bool NNPP::set_input_size(int inw, int inh)
{
	if (inw != this->inw || inh != this->inh) {
		this->inw = inw;
		this->inh = inh;
		set_param();
		return true;
	}
	return false;
}

bool NNPP::set_static_param(int outw, int outh, bool align_center, bool kar, bool reverse_channel,
	const float *mean, const float *var_recip)
{
	this->inw = 1920;
	this->inh = 1081;
	this->outw = outw;
	this->outh = outh;
	this->align_center = align_center;
	this->kar = kar;
	this->reverse_channel = reverse_channel;
	set_param();

	if (h_mean || h_var_recip || d_mean || d_var_recip) {
		LogError("normalization parameter has been initialized\n");
		return false;
	}
	
	h_mean.reset(new (std::nothrow) float[3]);
	if (!h_mean) {
		LogError("allocate memory failed\n");
		return false;
	}
	
	h_var_recip.reset(new (std::nothrow) float[3]);
	if (!h_var_recip) {
		LogError("allocate memory failed\n");
		return false;
	}
	
	for (int i = 0; i <3; ++i) {
		h_mean[i] = mean[i];
		h_var_recip[i] = var_recip[i];
	}

	d_mean.reset(reinterpret_cast<float *>(cudaMallocWC(3 * sizeof(float))), [](void *p){
		LogDebug("cudaFree d_mean\n");
		cudaFree(p);
	});
	if (!d_mean) {
		return false;
	}
	
	cudaError_t err = cudaMemcpy(d_mean.get(), h_mean.get(), 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		LogError("cudaMemcpy failed\n");
		return false;
	}

	d_var_recip.reset(reinterpret_cast<float *>(cudaMallocWC(3 * sizeof(float))), [](void *p){
		LogDebug("cudaFree d_var_recip\n");
		cudaFree(p);
	});
	if (!d_var_recip) {
		return false;
	}
	
	err = cudaMemcpy(d_var_recip.get(), h_var_recip.get(), 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		LogError("cudaMemcpy failed\n");
		return false;
	}
	
	return true;
}

void NNPP::forward(const uint8_t *d_in, float *d_out, cudaStream_t stream)
{
	static const int threads = 256;
	const int blocks = (w * h + threads - 1) / threads;
	cudaMemcpyAsync(d_mean.get(), h_mean.get(), 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_var_recip.get(), h_var_recip.get(), 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
	bilinear_interpolate<<<blocks, threads, 0, stream>>>((uchar3 *)d_in, d_out,
		inw, inh, outw, outh, w, h, padw, padh, 1 / scale, d_mean.get(), d_var_recip.get(), reverse_channel);
}

void NNPP::backward_inplace(std::vector<Detection> &dets, const std::vector<int> &mask) const
{
	for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
		if (!mask.empty() && std::find(mask.begin(), mask.end(), i) == mask.end())
			continue;
		auto &det = dets[i];
		det.box.x = static_cast<int>(roundf((det.box.x - padw) / scale));
		det.box.y = static_cast<int>(roundf((det.box.y - padh) / scale));
		det.box.width = static_cast<int>(roundf(det.box.width / scale));
		det.box.height = static_cast<int>(roundf(det.box.height / scale));
	}
}

NNPP::~NNPP()
{
}

void NNPP::set_param()
{
	if (kar) {
		float sw = outw / static_cast<float>(inw);
		float sh = outh / static_cast<float>(inh);
		scale = sw < sh ? sw : sh;
		w = static_cast<int>(roundf(scale * inw));
		h = static_cast<int>(roundf(scale * inh));
		if (align_center) {
			float pad_w_half = (outw - w) * .5f;
			float pad_h_half = (outh - h) * .5f;
			padw = static_cast<int>(roundf(pad_w_half - .1f));
			padh = static_cast<int>(roundf(pad_h_half - .1f));
		} else {
			padw = 0;
			padh = 0;
		}
		LogDebug("w=%d, h=%d, padw=%d, padh=%d\n", w, h, padw, padh);
	} else {
		w = outw;
		h = outh;
		padw = 0;
		padh = 0;
		// TODO: scale x, scale y
	}
}

} // namespace algorithm