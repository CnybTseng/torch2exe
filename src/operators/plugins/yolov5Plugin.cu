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
 * @brief YOLOV5 plugin module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cassert>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "utils/logger.h"
#include "yolov5Plugin.h"

// #define ENABLE_YOLOV5PLUTIN_FP16

namespace nvinfer1 {

__device__ float sigmoid(const float x)
{
	return 1.f / (1.f + expf(-x));
}

__device__ __half sigmoid(const __half x)
{
	__half a = __float2half(1.f);
	return a / (a + hexp(-x));
}

__global__ void decode(const float *input, float *output, const int *anchor, const int numAnchor, const int numClass,
	const int netWidth, const int netHeight, const int gridWidth, const int gridHeight, const int maxNumObj,
	const int outBatchStep, const int total, float ignoreThresh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > total) return;
	
	const int gridSize = gridWidth * gridHeight;
	const int batchIdx = idx / gridSize;
	const int gridIdx = idx - batchIdx * gridSize;
	const int gridY = gridIdx / gridWidth;
	const int gridX = gridIdx % gridWidth;
	const int dims = numClass + 5;
	const float *in = input + batchIdx * gridSize * numAnchor * dims;
	float *out = output + batchIdx * outBatchStep;
	
	// NCHW, [x,y,w,h,prob,classProb1,classProb2,...,classProbN]
	for (int i = 0; i < numAnchor; ++i) {
		// objectness
		float prob = sigmoid(in[i * gridSize * dims + 4 * gridSize + idx]);
		if (prob < ignoreThresh) continue;
	
		// update object count
		int count = atomicAdd((int *)out, 1);
		if (count >= maxNumObj) {
			count = atomicSub((int *)out, 1);
			return;
		}
		
		char *data = (char *)out + sizeof(float) + sizeof(algorithm::Detection) * count;
		algorithm::Detection *det = (algorithm::Detection *)data;
		
		// most likely class index and score
		int class_idx = 0;
		float max_class_prob = 0;
		for (int j = 0; j < numClass; ++j) {
			float class_prob = sigmoid(in[i * gridSize * dims + (5 + j) * gridSize + idx]);
			if (class_prob > max_class_prob) {
				max_class_prob = class_prob;
				class_idx = j;
			}
		}
		det->category = class_idx;
		det->score = prob * max_class_prob;
		
		// box
		float x = in[i * gridSize * dims + idx];
		float y = in[i * gridSize * dims + gridSize + idx];
		float w = in[i * gridSize * dims + gridSize * 2 + idx];
		float h = in[i * gridSize * dims + gridSize * 3 + idx];
		det->box.x = (2.f * sigmoid(x) - .5f + gridX) * netWidth / gridWidth;
		det->box.y = (2.f * sigmoid(y) - .5f + gridY) * netHeight / gridHeight;
		det->box.width = 2.f * sigmoid(w);
		det->box.width = det->box.width * det->box.width * anchor[2 * i];
		det->box.height = 2.f * sigmoid(h);
		det->box.height = det->box.height * det->box.height * anchor[2 * i + 1];
	}
}

__global__ void decode(const __half *input, float *output, const int *anchor, const int numAnchor, const int numClass,
	const int netWidth, const int netHeight, const int gridWidth, const int gridHeight, const int maxNumObj,
	const int outBatchStep, const int total, __half ignoreThresh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > total) return;

	const int gridSize = gridWidth * gridHeight;
	const int batchIdx = idx / gridSize;
	const int gridIdx = idx - batchIdx * gridSize;
	const int gridY = gridIdx / gridWidth;
	const int gridX = gridIdx % gridWidth;
	const int dims = numClass + 5;
	const __half *in = input + batchIdx * gridSize * numAnchor * dims;
	float *out = output + batchIdx * outBatchStep;
	
	// NCHW, [x,y,w,h,prob,classProb1,classProb2,...,classProbN]
	for (int i = 0; i < numAnchor; ++i) {
		// objectness
		__half prob = sigmoid(in[i * gridSize * dims + 4 * gridSize + idx]);
		if (__hlt(prob, ignoreThresh)) continue;

		// update object count
		int count = (int)atomicAdd(out, 1);
		if (count > maxNumObj) return;
		
		char *data = (char *)out + sizeof(float) + sizeof(algorithm::Detection) * count;
		algorithm::Detection *det = (algorithm::Detection *)data;
		
		// most likely class index and score
		int class_idx = 0;
		__half max_class_prob = 0;
		for (int j = 0; j < numClass; ++j) {
			__half class_prob = sigmoid(in[i * gridSize * dims + (5 + j) * gridSize + idx]);
			if (__hgt(class_prob, max_class_prob)) {
				max_class_prob = class_prob;
				class_idx = j;
			}
		}
		det->category = class_idx;
		det->score = __half2float(__hmul(prob, max_class_prob));

		// box
		__half x = in[i * gridSize * dims + idx];
		__half y = in[i * gridSize * dims + gridSize + idx];
		__half w = in[i * gridSize * dims + gridSize * 2 + idx];
		__half h = in[i * gridSize * dims + gridSize * 3 + idx];
		
		x = sigmoid(x);
		x = __hmul(__float2half(2.f), x);
		x = __hsub(x, __float2half(.5f));
		x = __hadd(x, __int2half_rz(gridX));
		x = __hmul(x, __int2half_rz(netWidth / gridWidth));
		det->box.x = __half2float(x);
		
		y = sigmoid(y);
		y = __hmul(__float2half(2.f), y);
		y = __hsub(y, __float2half(.5f));
		y = __hadd(y, __int2half_rz(gridY));
		y = __hmul(y, __int2half_rz(netHeight / gridHeight));
		det->box.y = __half2float(y);
		
		w = sigmoid(w);
		w = __hmul(__float2half(2.f), w);
		w = __hmul(w, w);
		w = __hmul(w, __int2half_rz(anchor[2 * i]));
		det->box.width = w;
		
		h = sigmoid(h);
		h = __hmul(__float2half(2.f), h);
		h = __hmul(h, h);
		h = __hmul(h, __int2half_rz(anchor[2 * i + 1]));
		det->box.width = h;
	}
}

YOLOV5Plugin::YOLOV5Plugin(const PluginFieldCollection& fc)
{
	mField = *((algorithm::YOLOV5PluginField *)fc.fields[0].data);
	for (int32_t i = 0; i < mField.num_anchor << 1; ++i) {
		LogDebug("anchor[%d] = %d\n", i, mField.anchor[i]);
	}
}

YOLOV5Plugin::YOLOV5Plugin(const void *data, size_t length)
{
	const char *d = reinterpret_cast<const char *>(data);
	const char* const a = d;
	mField.max_num_obj = algorithm::read<int32_t>(d);
	mField.down_sample_ratio = algorithm::read<int32_t>(d);
	mField.num_anchor = algorithm::read<int32_t>(d);
	for (int32_t i = 0; i < mField.num_anchor << 1; ++i) {
		mField.anchor[i] = algorithm::read<int32_t>(d);
		LogDebug("anchor[%d] = %d from byte stream\n", i, mField.anchor[i]);
	}
	mInputDims.nbDims = algorithm::read<int32_t>(d);
	for (int32_t i = 0; i < mInputDims.nbDims; ++i) {
		mInputDims.d[i] = algorithm::read<int32_t>(d);
	}
	mInputDataType = algorithm::read<DataType>(d);
	assert(d == a + length);
	(void)a;
}

YOLOV5Plugin::~YOLOV5Plugin()
{
}

//! This function is called by the builder prior to initialize()
void YOLOV5Plugin::configurePlugin(const PluginTensorDesc *in, int32_t nbInput, const PluginTensorDesc *out, int32_t nbOutput)
{
	assert(in && nbInput == 1);
	assert(out && nbOutput == 1);
	assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
	mInputDims = in[0].dims; // mInputDims will be available for serialize() and enqueue()
	mInputDataType = in[0].type;
}

//! The combination of kLINEAR + kHALF/kFLOAT is supported for input.
//! The combination of kLINEAR + kFLOAT is supported for output.
bool YOLOV5Plugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const
{
	assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
	bool condition = inOut[pos].format == TensorFormat::kLINEAR;
#ifdef ENABLE_YOLOV5PLUTIN_FP16
	condition &= (inOut[pos].type == (pos < 1 ? DataType::kFLOAT : DataType::kFLOAT)) ||
                 (inOut[pos].type == (pos < 1 ? DataType::kHALF  : DataType::kFLOAT));	
#else
	condition &= inOut[pos].type == DataType::kFLOAT; // both input and output are kFLOAT
#endif
	return condition;
}

DataType YOLOV5Plugin::getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const
{
	assert(inputTypes && nbInputs == 1);
	(void)index;
	return DataType::kFLOAT; // set kFLOAT data type for output op
}

bool YOLOV5Plugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted, int32_t nbInputs) const
{
	return false;
}

bool YOLOV5Plugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const
{
	return false;
}

IPluginV2Ext *YOLOV5Plugin::clone() const
{
	auto *plugin = new YOLOV5Plugin(*this);
    return plugin;
}

const char *YOLOV5Plugin::getPluginType() const
{
	return "YOLOV5";
}

const char *YOLOV5Plugin::getPluginVersion() const
{
	return "1";
}

int32_t YOLOV5Plugin::getNbOutputs() const
{
	return 1;
}

Dims YOLOV5Plugin::getOutputDimensions(int32_t index, const Dims *inputs, int32_t nbInputDims)
{
	assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 4);
	int32_t d = mField.max_num_obj * sizeof(algorithm::Detection) / sizeof(float);
	return Dims4(inputs[0].d[0], d + 1, 1, 1);
}

int32_t YOLOV5Plugin::initialize()
{
	LogDebug("YOLOV5Plugin::initialize\n");
	const size_t size = (mField.num_anchor << 1) * sizeof(int);
	cudaError_t err = cudaMalloc(&mAnchor, size);
	if (err != cudaSuccess) {
		LogError("cudaMalloc failed\n");
		return -1;
	}
	
	err = cudaMemcpy(mAnchor, mField.anchor, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		LogError("cudaMemcpy failed\n");
		return -1;
	}
	return 0;
}

void YOLOV5Plugin::terminate()
{
	LogDebug("YOLOV5Plugin::terminate\n");
	cudaFree(mAnchor);
}

size_t YOLOV5Plugin::getWorkspaceSize(int32_t maxBatchSize) const
{
	return 0;
}

int32_t YOLOV5Plugin::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
	if (mInputDataType == DataType::kFLOAT) {
		LogDebug("inference with kFLOAT\n");
		const float *const input = reinterpret_cast<const float *const>(inputs[0]);
		float *output = reinterpret_cast<float *>(outputs[0]);

		const size_t step = mField.max_num_obj * sizeof(algorithm::Detection) / sizeof(float) + 1;
		const int numClass = mInputDims.d[1] / mField.num_anchor - 5;
		const int netWidth = mInputDims.d[3] * mField.down_sample_ratio;
		const int netHeight = mInputDims.d[2] * mField.down_sample_ratio;
		const float ignoreThresh = 0.1;

		const int32_t total = mInputDims.d[0] * mInputDims.d[2] * mInputDims.d[3];
		int32_t threads = total >= 256 ? 256 : total;
		const int32_t blocks = (total + threads - 1) / threads;
		decode<<<blocks, threads, 0, stream>>>(input, output, mAnchor, mField.num_anchor, numClass, netWidth,
			netHeight, mInputDims.d[3], mInputDims.d[2], mField.max_num_obj, step, total, ignoreThresh
		);
	} else if (mInputDataType == DataType::kHALF) {
		LogDebug("inference with kHALF\n");
		const __half *const input = reinterpret_cast<const __half *const>(inputs[0]);
		float *output = reinterpret_cast<float *>(outputs[0]);

		const size_t step = mField.max_num_obj * sizeof(algorithm::Detection) / sizeof(float) + 1;
		const int numClass = mInputDims.d[1] / mField.num_anchor - 5;
		const int netWidth = mInputDims.d[3] * mField.down_sample_ratio;
		const int netHeight = mInputDims.d[2] * mField.down_sample_ratio;
		const __half ignoreThresh = __float2half(.1f);

		const int32_t total = mInputDims.d[0] * mInputDims.d[2] * mInputDims.d[3];
		int32_t threads = total >= 256 ? 256 : total;
		const int32_t blocks = (total + threads - 1) / threads;
		decode<<<blocks, threads, 0, stream>>>(input, output, mAnchor, mField.num_anchor, numClass, netWidth,
			netHeight, mInputDims.d[3], mInputDims.d[2], mField.max_num_obj, step, total, ignoreThresh
		);
	}
	return 0;
}

size_t YOLOV5Plugin::getSerializationSize() const
{
	size_t serializationSize = 0;
	serializationSize += sizeof(mField.max_num_obj);
	serializationSize += sizeof(mField.down_sample_ratio);
	serializationSize += sizeof(mField.num_anchor);
	serializationSize += (mField.num_anchor * sizeof(mField.anchor[0]) << 1);
	serializationSize += sizeof(mInputDims.nbDims);
	serializationSize += mInputDims.nbDims * sizeof(mInputDims.d[0]);
	serializationSize += sizeof(DataType);
	return serializationSize;
}

void YOLOV5Plugin::serialize(void *buffer) const
{
	char* d = static_cast<char*>(buffer);
    const char* const a = d;
	algorithm::write(d, mField.max_num_obj);
	algorithm::write(d, mField.down_sample_ratio);
	algorithm::write(d, mField.num_anchor);
	for (int32_t i = 0; i < (mField.num_anchor << 1); ++i) {
		algorithm::write(d, mField.anchor[i]);
	}
	algorithm::write(d, mInputDims.nbDims);
	for (int32_t i = 0; i < mInputDims.nbDims; ++i) {
		algorithm::write(d, mInputDims.d[i]);
	}
	algorithm::write(d, mInputDataType);
	assert(d == a + getSerializationSize());
	(void)a;
}

void YOLOV5Plugin::destroy()
{
	delete this;
}

void YOLOV5Plugin::setPluginNamespace(const char *pluginNamespace)
{
	mNamespace = pluginNamespace;
}

const char *YOLOV5Plugin::getPluginNamespace() const
{
	return mNamespace.data();
}

const char *YOLOV5PluginCreator::getPluginName() const
{
	return "YOLOV5";
}

const char *YOLOV5PluginCreator::getPluginVersion() const
{
	return "1";
}

const PluginFieldCollection *YOLOV5PluginCreator::getFieldNames()
{
	return &mFieldCollection;
}

IPluginV2 *YOLOV5PluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
	if (1 != fc->nbFields || !fc->fields) return nullptr;
	mPluginName = name;
	mFieldCollection = *fc;
	auto plugin = new YOLOV5Plugin(*fc);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

IPluginV2 *YOLOV5PluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
	mPluginName = name;
	auto plugin = new YOLOV5Plugin(serialData, serialLength);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

void YOLOV5PluginCreator::setPluginNamespace(const char *pluginNamespace)
{
	mNamespace = pluginNamespace;
}

const char *YOLOV5PluginCreator::getPluginNamespace() const
{
	return mNamespace.c_str();
}

} // namespace nvinfer1