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
 * @brief Algorithm interface module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cstdarg>

#include <cuda_runtime_api.h>

#include "algorithm.h"
#include "utils/algorithmFactory.h"
#include "utils/cuda.h"
#include "utils/deleter.h"
#include "utils/logger.h"

namespace algorithm {

std::shared_ptr<Image> Image::alloc(uint16_t h, uint16_t w, uint16_t s, bool pinned)
{
	std::shared_ptr<Image> image;
	const size_t size = h * s + sizeof(Image);
	if (pinned) {
		image.reset(reinterpret_cast<Image *>(cudaMallocHostWC(size)), cudaFreerHost("image"));
	} else {
		image.reset(reinterpret_cast<Image *>(new (std::nothrow)char[size]), ArrayDeleter("image"));
	}
	if (image) {
		image->width = w;
		image->height = h;
		image->stride = s;
	}
	return image;
}

std::unique_ptr<Algorithm, Deleter> Algorithm::create(const char *name)
{
	return AlgorithmFactory<Algorithm>::get_algorithm(name);
}

void Algorithm::register_logger(LogCallback fun)
{
	logging::register_logger(fun);
}

AlgorithmListSP Algorithm::get_algorithm_list()
{
	return AlgorithmFactory<Algorithm>::get_algorithm_list();
}

} // namespace algorithm