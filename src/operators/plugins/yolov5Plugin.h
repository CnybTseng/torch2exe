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

#ifndef YOLOV5PLUGIN_H_
#define YOLOV5PLUGIN_H_

#include <string>

#include <NvInfer.h>

#include "algorithm.h"

namespace algorithm {

static constexpr int32_t MAX_NUM_ANCHOR = 10;

struct YOLOV5PluginField
{
	int32_t max_num_obj;					// maximum number of objects
	int32_t down_sample_ratio;				// 8, 16, 32, 64
	int32_t num_anchor;						// number of anchors per plugin, e.g, 3
	int32_t anchor[MAX_NUM_ANCHOR << 1];	// anchor array
};

} // namespace algorithm

namespace nvinfer1 {

class YOLOV5Plugin : public IPluginV2IOExt
{
public:
	YOLOV5Plugin() = delete;
	YOLOV5Plugin(const PluginFieldCollection& fc);
	YOLOV5Plugin(const void *data, size_t length);
	virtual ~YOLOV5Plugin();
	// inherited from nvinfer1::IPluginV2IOExt
	virtual void configurePlugin(const PluginTensorDesc *in, int32_t nbInput, const PluginTensorDesc *out, int32_t nbOutput) override;
	using IPluginV2Ext::configurePlugin; // solve the `... virtual function override intended?` warning
	virtual bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const override;
	// inherited from nvinfer1::IPluginV2Ext
	virtual DataType getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const override;
	virtual bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted, int32_t nbInputs) const override;
	virtual bool canBroadcastInputAcrossBatch(int32_t inputIndex) const override;
	virtual IPluginV2Ext *clone() const override;
	// inherited from nvinfer1::IPluginV2
	virtual const char *getPluginType() const override;
	virtual const char *getPluginVersion() const override;
	virtual int32_t getNbOutputs() const override;
	virtual Dims getOutputDimensions(int32_t index, const Dims *inputs, int32_t nbInputDims) override;
	virtual int32_t initialize() override;
	virtual void terminate() override;
	virtual size_t getWorkspaceSize(int32_t maxBatchSize) const override;
	virtual int32_t enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
	virtual size_t getSerializationSize() const override;
	virtual void serialize(void *buffer) const override;
	virtual void destroy() override;
	virtual void setPluginNamespace(const char *pluginNamespace) override;
	virtual const char *getPluginNamespace() const override;
private:
	std::string mNamespace;
	algorithm::YOLOV5PluginField mField;
	Dims mInputDims;
	DataType mInputDataType;
	int *mAnchor;
};

class YOLOV5PluginCreator : public IPluginCreator
{
public:
	YOLOV5PluginCreator() = default;
	virtual ~YOLOV5PluginCreator() override = default;
	virtual const char *getPluginName() const override;
	virtual const char *getPluginVersion() const override;
	virtual const PluginFieldCollection *getFieldNames() override;
	virtual IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override;
	virtual IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;
	virtual void setPluginNamespace(const char *pluginNamespace) override;
	virtual const char *getPluginNamespace() const override;
private:
	std::string mNamespace;
    std::string mPluginName;
	PluginFieldCollection mFieldCollection{0, nullptr};
};
REGISTER_TENSORRT_PLUGIN(YOLOV5PluginCreator);

} // namespace nvinfer1

#endif // namespace YOLOV5PLUGIN_H_