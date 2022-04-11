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
 * @brief YOLOX.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifndef YOLOX_H_
#define YOLOX_H_

#include <memory>
#include <string>
#include <vector>

#include "algorithm.h"
#include "detector.h"
#include "operators/nnpp.h"
#include "nnEngine.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace detector {

class YOLOX : public Detector, public RegisteredInFactory<Algorithm, YOLOX>
{
public:
	YOLOX() = default;
	static std::unique_ptr<Algorithm, Deleter> create(const char *cfg);
	static std::string get_name(void);
	virtual ~YOLOX() = default;
	virtual bool init(const char *cfg) override;
	virtual bool execute(const BlobSP &input, BlobSP &output) override;
	virtual void destroy(void) override;
private:
	NNPP nnpp;
};

} // namespace detector
} // namespace algorithm

#endif // YOLOX_H_