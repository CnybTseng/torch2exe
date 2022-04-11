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
 * @brief Neural network engine.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifndef NNENGINE_H_
#define NNENGINE_H_

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <logging.h>

#include "algorithm.h"
#include "utils/configurer.h"

namespace algorithm {
namespace tensorrt {

/**
 * @class NNEngine
 * @brief 神经网络引擎.
 */
class NNEngine
{
public:
	NNEngine() = default;
	virtual ~NNEngine() = default;
	
	/**
	 * @brief 构建神经网络引擎.
	 * @param cfg 神经网络模型配置文件.
	 * @return 神经网络引擎构建状态. 成功: true, 失败: false.
	 */
	virtual bool build(const char *cfg);
	virtual bool execute(void **bindings, int batch_size, bool block=false);
	virtual void destroy(void);
	virtual const cudaStream_t &get_stream() const {return stream;}
	virtual const std::vector<int> &get_binding_shape(int index) const;
	static sample::Logger logger;		///! 日志接口
private:
	virtual bool create_context(const char *engine_path);
	virtual bool set_binding_shape(const char **id, size_t id_count, const Configurer &graph);
	int batch_size;
	cudaStream_t stream;							///! CUDA流
	nvinfer1::IRuntime *rt{nullptr};				///! TensorRT运行时
	nvinfer1::ICudaEngine *cue{nullptr};			///! TensorRT的CUDA引擎
	nvinfer1::IExecutionContext *ctx{nullptr};		///! TensorRT的执行上下文
	std::vector<std::vector<int>> binding_shape;
};

} // namespace tensorrt
} // namespace algorithm

#endif // NNENGINE_H_
