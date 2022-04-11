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
 * @brief Smart pointer deleter module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cuda_runtime_api.h>

#include "utils/deleter.h"
#include "utils/logger.h"

namespace algorithm {

void cudaFreer::operator()(void *obj) const
{
	LogDebug("cudaFree %s\n", signature.c_str());
	cudaFree(obj);
}

void cudaFreerHost::operator()(void *obj) const
{
	LogDebug("cudaFreeHost %s\n", signature.c_str());
	if (obj) {
		cudaFreeHost(obj);
	}
}

void ArrayDeleter::operator()(void *obj) const
{
	LogDebug("delete [] %s\n", signature.c_str());
	if (obj) {
		delete [] obj;
	}
}

void cudafreer(void *p)
{
	LogDebug("cudaFree\n");
	cudaFree(p);
}

void cudafreerhost(void *p)
{
	LogDebug("cudaFreeHost\n");
	if (p) {
		cudaFreeHost(p);
	}
}

void array_deleter(void *p)
{
	LogDebug("delete []\n");
	if (p) {
		delete [] p;
	}
}

} // namespace algorithm