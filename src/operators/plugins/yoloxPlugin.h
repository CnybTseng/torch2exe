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

#ifndef YOLOXPLUGIN_H_
#define YOLOXPLUGIN_H_

#include <string>

#include <NvInfer.h>

#include "algorithm.h"

namespace algorithm {

struct YOLOXPluginField
{
	int32_t max_num_obj;					// maximum number of objects
	int32_t down_sample_ratio;				// 8, 16, 32
};

} // namespace algorithm

namespace nvinfer1 {

class YOLOXPlugin : public IPluginV2IOExt
{
public:
	YOLOXPlugin() = delete;
	YOLOXPlugin(const PluginFieldCollection& fc);
	YOLOXPlugin(const void *data, size_t length);
	virtual ~YOLOXPlugin();
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
	algorithm::YOLOXPluginField mField;
	Dims mInputDims;
	DataType mInputDataType;
};

class YOLOXPluginCreator : public IPluginCreator
{
public:
	YOLOXPluginCreator() = default;
	virtual ~YOLOXPluginCreator() override = default;
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
REGISTER_TENSORRT_PLUGIN(YOLOXPluginCreator);

} // namespace nvinfer1

#endif // namespace YOLOXPLUGIN_H_