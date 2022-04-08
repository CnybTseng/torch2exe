#include <memory>
#include <vector>

#include <NvInfer.h>

#include "yolov5.h"
#include "plugins/yolov5Plugin.h"
#include "utils/configurer.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

std::unique_ptr<Operator, Deleter> YOLOV5::create(const char *name)
{
	return std::unique_ptr<Operator, Deleter>(new YOLOV5, [](void *p){((Operator*)p)->destroy();});
}

std::string YOLOV5::get_name(void)
{
	return "YOLOV5";
}

bool YOLOV5::set(const char *id, const Json::Value &cfg, NetworkContext &ctx)
{
	const Configurer opt(cfg);
	const std::vector<std::string> inm = opt.get("input", std::vector<std::string>());
	if (1 != inm.size()) {
		LogWarn("wrong number of inputs for operator %s: %d\n", id, inm.size());
		return false;
	}

	auto iter = ctx.output.find(inm[0]);
	if (iter == ctx.output.end()) {
		LogError("invalid input for operator %s: %s\n", id, inm[0]);
		return false;
	}

	nvinfer1::ITensor *input = iter->second[0];
	const std::vector<int> anchor = opt.get("anchor", std::vector<int>());
	if (6 != anchor.size()) {
		LogError("invalid anchor size for operator %s: %d\n", id, anchor.size());
		return false;
	}
	
	const int max_num_obj = opt.get("max_num_obj", int(1000));
	const int down_sample_ratio = opt.get("down_sample_ratio", int(0));
	if (0 == down_sample_ratio) {
		LogError("invalid down sample ratio for operator %s: %d\n", id, down_sample_ratio);
		return false;
	}
	
	YOLOV5PluginField data;
	data.max_num_obj = max_num_obj;
	data.down_sample_ratio = down_sample_ratio;
	data.num_anchor = static_cast<int32_t>(anchor.size() >> 1);
	memcpy(data.anchor, anchor.data(), anchor.size() * sizeof(int32_t));
	
	static const int32_t nbFields = 1;
	nvinfer1::PluginField fields[nbFields];
	fields[0].name = "YOLOV5PluginField";
	fields[0].data = &data;
	fields[0].type = nvinfer1::PluginFieldType::kINT32;
	fields[0].length = static_cast<int32_t>(sizeof(data));
	
	auto creator = getPluginRegistry()->getPluginCreator("YOLOV5", "1");
	nvinfer1::PluginFieldCollection pfc;
	pfc.nbFields = nbFields;
	pfc.fields = fields;
	std::shared_ptr<nvinfer1::IPluginV2> plugin(
		creator->createPlugin(id, &pfc), [](void *p){
			LogDebug("destroy YOLOV5 plugin\n");
			((nvinfer1::IPluginV2 *)p)->destroy();
	});
	
	nvinfer1::ITensor *inputs[] = {input};
	static const int32_t nbInputs = 1;
	auto op = ctx.network->addPluginV2(inputs, nbInputs, *plugin.get());
	if (!op) {
		LogError("addPluginV2 for operator %s failed\n", id);
		return false;
	}
	
	// keep the plugin alive before engine generated
	ctx.plugins.emplace_back(plugin);

	op->setName(id);
	op->getOutput(0)->setName(id);
	ctx.output[id] = {op->getOutput(0)};
	return true;
}

} // namespace tensorrt
} // namespace algorithm