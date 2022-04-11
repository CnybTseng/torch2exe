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
 * @brief YOLOX plugin module.
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
#include "yoloxPlugin.h"

// #define ENABLE_YOLOV5PLUTIN_FP16

namespace nvinfer1 {

__global__ void decode(const float *input, float *output, const int numClass,
	const int netWidth, const int netHeight, const int gridWidth, const int gridHeight,
	const int maxNumObj, const int outBatchStep, const int total, float ignoreThresh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > total) return;
	
	const int gridSize = gridWidth * gridHeight;
	const int batchIdx = idx / gridSize;
	const int gridIdx = idx - batchIdx * gridSize;
	const int gridY = gridIdx / gridWidth;
	const int gridX = gridIdx % gridWidth;
	const int dims = numClass + 5;
	const float *in = input + batchIdx * gridSize * dims;
	float *out = output + batchIdx * outBatchStep;

	// objectness
	float prob = in[4 * gridSize + idx];
	if (prob < ignoreThresh) return;

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
		float class_prob = in[(5 + j) * gridSize + idx];
		if (class_prob > max_class_prob) {
			max_class_prob = class_prob;
			class_idx = j;
		}
	}
	det->category = class_idx;
	det->score = prob * max_class_prob;
	
	// box
	det->box.x = (in[idx] + gridX) * netWidth / gridWidth;
	det->box.y = (in[gridSize + idx] + gridY) * netHeight / gridHeight;
	det->box.width = expf(in[gridSize * 2 + idx]) * netWidth / gridWidth;
	det->box.height = expf(in[gridSize * 3 + idx]) * netHeight / gridHeight;
}

YOLOXPlugin::YOLOXPlugin(const PluginFieldCollection& fc)
{
	mField = *((algorithm::YOLOXPluginField *)fc.fields[0].data);
}

YOLOXPlugin::YOLOXPlugin(const void *data, size_t length)
{
	const char *d = reinterpret_cast<const char *>(data);
	const char* const a = d;
	mField.max_num_obj = algorithm::read<int32_t>(d);
	mField.down_sample_ratio = algorithm::read<int32_t>(d);
	mInputDims.nbDims = algorithm::read<int32_t>(d);
	for (int32_t i = 0; i < mInputDims.nbDims; ++i) {
		mInputDims.d[i] = algorithm::read<int32_t>(d);
	}
	mInputDataType = algorithm::read<DataType>(d);
	assert(d == a + length);
	(void)a;
}

YOLOXPlugin::~YOLOXPlugin()
{
}

//! This function is called by the builder prior to initialize()
void YOLOXPlugin::configurePlugin(const PluginTensorDesc *in, int32_t nbInput, const PluginTensorDesc *out, int32_t nbOutput)
{
	assert(in && nbInput == 1);
	assert(out && nbOutput == 1);
	assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
	mInputDims = in[0].dims; // mInputDims will be available for serialize() and enqueue()
	mInputDataType = in[0].type;
}

//! The combination of kLINEAR + kHALF/kFLOAT is supported for input.
//! The combination of kLINEAR + kFLOAT is supported for output.
bool YOLOXPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const
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

DataType YOLOXPlugin::getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const
{
	assert(inputTypes && nbInputs == 1);
	(void)index;
	return DataType::kFLOAT; // set kFLOAT data type for output op
}

bool YOLOXPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted, int32_t nbInputs) const
{
	return false;
}

bool YOLOXPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const
{
	return false;
}

IPluginV2Ext *YOLOXPlugin::clone() const
{
	auto *plugin = new YOLOXPlugin(*this);
    return plugin;
}

const char *YOLOXPlugin::getPluginType() const
{
	return "YOLOX";
}

const char *YOLOXPlugin::getPluginVersion() const
{
	return "1";
}

int32_t YOLOXPlugin::getNbOutputs() const
{
	return 1;
}

Dims YOLOXPlugin::getOutputDimensions(int32_t index, const Dims *inputs, int32_t nbInputDims)
{
	assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 4);
	int32_t d = mField.max_num_obj * sizeof(algorithm::Detection) / sizeof(float);
	return Dims4(inputs[0].d[0], d + 1, 1, 1);
}

int32_t YOLOXPlugin::initialize()
{
	LogDebug("YOLOXPlugin::initialize\n");
	return 0;
}

void YOLOXPlugin::terminate()
{
	LogDebug("YOLOXPlugin::terminate\n");
}

size_t YOLOXPlugin::getWorkspaceSize(int32_t maxBatchSize) const
{
	return 0;
}

int32_t YOLOXPlugin::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
	if (mInputDataType == DataType::kFLOAT) {
		LogDebug("inference with kFLOAT\n");
		const float *const input = reinterpret_cast<const float *const>(inputs[0]);
		float *output = reinterpret_cast<float *>(outputs[0]);

		const size_t step = mField.max_num_obj * sizeof(algorithm::Detection) / sizeof(float) + 1;
		const int numClass = mInputDims.d[1] - 5;
		const int netWidth = mInputDims.d[3] * mField.down_sample_ratio;
		const int netHeight = mInputDims.d[2] * mField.down_sample_ratio;
		const float ignoreThresh = 0.1;

		const int32_t total = mInputDims.d[0] * mInputDims.d[2] * mInputDims.d[3];
		int32_t threads = total >= 256 ? 256 : total;
		const int32_t blocks = (total + threads - 1) / threads;
		decode<<<blocks, threads, 0, stream>>>(input, output, numClass, netWidth, netHeight,
			mInputDims.d[3], mInputDims.d[2], mField.max_num_obj, step, total, ignoreThresh
		);
	} else if (mInputDataType == DataType::kHALF) {
		;
	}
	return 0;
}

size_t YOLOXPlugin::getSerializationSize() const
{
	size_t serializationSize = 0;
	serializationSize += sizeof(mField.max_num_obj);
	serializationSize += sizeof(mField.down_sample_ratio);
	serializationSize += sizeof(mInputDims.nbDims);
	serializationSize += mInputDims.nbDims * sizeof(mInputDims.d[0]);
	serializationSize += sizeof(DataType);
	return serializationSize;
}

void YOLOXPlugin::serialize(void *buffer) const
{
	char* d = static_cast<char*>(buffer);
    const char* const a = d;
	algorithm::write(d, mField.max_num_obj);
	algorithm::write(d, mField.down_sample_ratio);
	algorithm::write(d, mInputDims.nbDims);
	for (int32_t i = 0; i < mInputDims.nbDims; ++i) {
		algorithm::write(d, mInputDims.d[i]);
	}
	algorithm::write(d, mInputDataType);
	assert(d == a + getSerializationSize());
	(void)a;
}

void YOLOXPlugin::destroy()
{
	delete this;
}

void YOLOXPlugin::setPluginNamespace(const char *pluginNamespace)
{
	mNamespace = pluginNamespace;
}

const char *YOLOXPlugin::getPluginNamespace() const
{
	return mNamespace.data();
}

const char *YOLOXPluginCreator::getPluginName() const
{
	return "YOLOX";
}

const char *YOLOXPluginCreator::getPluginVersion() const
{
	return "1";
}

const PluginFieldCollection *YOLOXPluginCreator::getFieldNames()
{
	return &mFieldCollection;
}

IPluginV2 *YOLOXPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
	if (1 != fc->nbFields || !fc->fields) return nullptr;
	mPluginName = name;
	mFieldCollection = *fc;
	auto plugin = new YOLOXPlugin(*fc);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

IPluginV2 *YOLOXPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
	mPluginName = name;
	auto plugin = new YOLOXPlugin(serialData, serialLength);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}

void YOLOXPluginCreator::setPluginNamespace(const char *pluginNamespace)
{
	mNamespace = pluginNamespace;
}

const char *YOLOXPluginCreator::getPluginNamespace() const
{
	return mNamespace.c_str();
}

} // namespace nvinfer1