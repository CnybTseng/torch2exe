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
 * @brief CUDA functional module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cstring>

#include <cuda_runtime_api.h>

#include "utils/cuda.h"
#include "utils/logger.h"

namespace algorithm {

void *cudaMallocWC(size_t size)
{
	void *ptr = nullptr;
	cudaError_t err = cudaMalloc(&ptr, size);
	if (cudaSuccess != err) {
		LogError("cuda malloc host failed\n");
		return nullptr;
	}
	return ptr;
}

void *cudaMallocHostWC(size_t size)
{
	void *ptr = nullptr;
	cudaError_t err = cudaMallocHost(&ptr, size);
	if (cudaSuccess != err) {
		LogError("cuda malloc host failed\n");
		return nullptr;
	}
	return ptr;
}

std::string cudaDeviceName(int device)
{
	cudaDeviceProp prop;
	cudaError_t err = cudaGetDeviceProperties(&prop, device);
	if (err != cudaSuccess) {
		LogError("cudaGetDeviceProperties failed\n");
		return "";
	}
	
	for (size_t i = 0; i < strlen(prop.name); ++i) {
		if (32 == prop.name[i]) {
			prop.name[i] = '-';
		} else if (prop.name[i] > 64 && prop.name[i] < 91) {
			prop.name[i] += 32;
		}
	}
	
	return std::string(prop.name);
}

}