#include <fstream>
#include <memory>
#include <vector>

#include <common.h>
#include <json/json.h>

#include "nnEngine.h"
#include "operator.h"
#include "platforms/library.h"
#include "platforms/path.h"

#include "utils/algorithmFactory.h"
#include "utils/cuda.h"
#include "utils/logger.h"

namespace algorithm {
namespace tensorrt {

static char *get_graph(const std::shared_ptr<void> &dll)
{
	char *graph_str = (char *)get_dlsym(dll.get(), "graph");
	if (!graph_str) {
		LogError("get_dlsym failed: graph\n");
		return nullptr;
	}
	return graph_str;
}

static char **get_id(const std::shared_ptr<void> &dll, size_t &id_count)
{
	char **id = (char **)get_dlsym(dll.get(), "id");
	if (!id) {
		LogError("get_dlsym failed: id\n");
		return nullptr;
	}
	
	size_t *id_count_ptr = (size_t *)get_dlsym(dll.get(), "id_count");
	if (!id_count_ptr) {
		LogError("get_dlsym failed: id_count\n");
		return nullptr;
	}

	id_count = *id_count_ptr;
	return id;
}

static float *get_weight(const std::shared_ptr<void> &dll, size_t &weight_count)
{
	float *weight = (float *)get_dlsym(dll.get(), "weight");
	if (!weight) {
		LogError("get_dlsym failed: weight\n");
		return nullptr;
	}
	
	size_t *weight_count_ptr = (size_t *)get_dlsym(dll.get(), "weight_count");
	if (!weight_count_ptr) {
		LogError("get_dlsym failed: weight_count\n");
		return nullptr;
	}

	weight_count = *weight_count_ptr;
	return weight;
}

//! kINTERNAL_ERROR, kERROR, kWARNING, kINFO, kVERBOSE
sample::Logger NNEngine::logger(nvinfer1::ILogger::Severity::kVERBOSE);

bool NNEngine::build(const char *cfg)
{	
	Configurer opt;
	if (!opt.init(cfg)) {
		LogError("NNEngine::build: Configurer::init failed\n");
		return false;
	}

	batch_size = opt.get("batch_size", int(1));
	std::string model_path = opt.get("id_weight_path", std::string(""));
	LogDebug("id_weight_path: %s\n", model_path.c_str());
	if (model_path == std::string("")) {
		LogError("get id_weight_path failed\n");
		return false;
	}
	
	std::shared_ptr<void> dll(
		open_dll(model_path.c_str()), [](void *p){
			LogDebug("close_dll\n");
			close_dll(p);
		}
	);
	if (!dll) {
		LogError("open_dll failed\n");
		return false;
	}
	
	size_t id_count = 0;
	char **id = get_id(dll, id_count);
	if (!id) return false;
	
	size_t weight_count = 0;
	float *weight = get_weight(dll, weight_count);
	if (!weight) return false;

	Configurer graph_opt;
	std::string graph_path = opt.get("graph_path", std::string(""));
	LogDebug("graph_path: %s\n", graph_path.c_str());
	if (graph_path != "" && file_exists(graph_path.c_str())) { // read from json file
		if (!graph_opt.init(graph_path.c_str())) {
			LogError("NNEngine::build: Configurer::init failed\n");
			return false;
		}
	} else { // read from library
		char *graph_str = get_graph(dll);
		if (!graph_str) return false;
		if (!graph_opt.init(graph_str, strlen(graph_str))) {
			LogError("NNEngine::build: Configurer::init failed\n");
			return false;
		}
	}
	
	bool use_int8 = opt.get("use_int8", false);
	bool use_fp16 = opt.get("use_fp16", false);
	std::string prec = "fp32.";
	if (use_int8) {
		prec = "int8.";
	} else if (use_fp16) {
		prec = "fp16.";
	}
	
	int device_id = opt.get("device", 0);
	std::string dev = cudaDeviceName(device_id);
	dev = dev != "" ? dev + "." : "";
	const std::string engine_path = replace_filename_suffix(model_path, dev + prec + "engine");
	if (!file_exists(engine_path.c_str())) {
		LogDebug("%s isn't exists, build it first\n", engine_path.c_str());
		std::unique_ptr<nvinfer1::IBuilder, Deleter> builder(
			nvinfer1::createInferBuilder(logger), [](void *p){
				LogDebug("nvinfer1::IBuilder::destroy\n");
				((nvinfer1::IBuilder *)p)->destroy();
			}
		);
		builder->setMaxBatchSize(batch_size);
		
		std::unique_ptr<nvinfer1::IBuilderConfig, Deleter> config(
			builder->createBuilderConfig(), [](void *p){
				LogDebug("nvinfer1::IBuilderConfig::destroy\n");
				((nvinfer1::IBuilderConfig *)p)->destroy();
			}
		);
		config->setMaxWorkspaceSize(16_MiB);
		if (use_int8 && builder->platformHasFastInt8()) {
			config->setFlag(nvinfer1::BuilderFlag::kINT8);
			config->setInt8Calibrator(nullptr); // TODO: set int8 calibrator!!!
		} else if (use_fp16 && builder->platformHasFastFp16()) {
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}

		NetworkContext ctx;
		ctx.weight = weight;
		ctx.len_read = 0;
		ctx.network.reset(
			builder->createNetworkV2(0U), [](void *p){
				LogDebug("nvinfer1::INetworkDefinition::destroy\n");
				((nvinfer1::INetworkDefinition *)p)->destroy();
			}
		);
		
		for (size_t i = 0; i < id_count; ++i) {
			Json::Value val;
			if (!graph_opt.get(id[i], val)) {
				LogError("get operator %s failed\n", id[i]);
				return false;
			}
			const Configurer ops_opt(val);
			const std::string name = ops_opt.get("name", std::string(""));
			LogDebug("%s %s\n", id[i], name.c_str());
			auto ops = AlgorithmFactory<Operator>::get_algorithm(name.c_str());
			if (!ops) {
				LogError("operator %s:%s not supported yet\n", id[i], name.c_str());
				return false;
			}
			if (!ops->set(id[i], val, ctx)) {
				LogError("set operator %s:%s failed\n", id[i], name.c_str());
				return false;
			}
		}

		if (weight_count != ctx.len_read) {
			LogError("weight_count: %d, len_read: %d\n", weight_count, ctx.len_read);
			return false;
		}

		std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine(
			builder->buildEngineWithConfig(*ctx.network.get(), *config.get()), [](void *p){
				LogDebug("nvinfer1::ICudaEngine::destroy\n");
				((nvinfer1::ICudaEngine *)p)->destroy();
			}
		);
		if (!engine) {
			LogError("IBuilder::buildEngineWithConfig failed\n");
			return false;
		}
		
		std::unique_ptr<nvinfer1::IHostMemory, Deleter> hmem(
			engine->serialize(), [](void *p){
				LogDebug("nvinfer1::IHostMemory::destroy\n");
				((nvinfer1::IHostMemory *)p)->destroy();
			}
		);
		if (!hmem) {
			LogError("ICudaEngine::serialize failed\n");
			return false;
		}

		std::ofstream ofs(engine_path, std::ios::binary);
		if (!ofs) {
			LogError("open engine file filed\n");
			return false;
		}
		
		ofs.write(reinterpret_cast<const char*>(hmem->data()), hmem->size());
		ofs.close();
		LogDebug("build %s done\n", engine_path.c_str());
	}

	if (!create_context(engine_path.c_str())) return false;
	return set_binding_shape((const char **)id, id_count, graph_opt);
}

bool NNEngine::execute(void **bindings, int batch_size, bool block)
{
	if (batch_size != this->batch_size) {
		LogError("wrong batch size, expect %d, but got %d\n", this->batch_size, batch_size);
		return false;
	}

	bool status = ctx->enqueue(batch_size, bindings, stream, nullptr);
	if (status && block) {
		cudaStreamSynchronize(stream);
	}
	return status;
}

void NNEngine::destroy(void)
{
	LogDebug("NNEngine::destroy\n");
	if (ctx) ctx->destroy();
	if (cue) cue->destroy();
	if (rt) rt->destroy();
	cudaStreamDestroy(stream);
}

const std::vector<int> &NNEngine::get_binding_shape(int index) const
{
	if (index > binding_shape.size()) {
		LogError("wrong index for binding_shape: %d\n", index);
		return std::vector<int>();
	}
	return binding_shape[index];
}

bool NNEngine::create_context(const char *engine_path)
{
	std::ifstream ifs(engine_path, std::ios::binary);
	if (!ifs.good()) {
		LogError("open %s failed\n", engine_path);
		return false;
	}
	
	ifs.seekg(0, ifs.end);
	const size_t size = ifs.tellg();
	ifs.seekg(0, ifs.beg);
	
	std::shared_ptr<char[]> blob(new (std::nothrow) char[size], [](char *p){
		LogDebug("delete [] blob\n");
		delete [] p;
	});
	if (!blob) {
		LogError("allocate memory failed\n");
		ifs.close();
		return false;
	}
	
	ifs.read(blob.get(), size);
	ifs.close();
	
	cudaError_t err = cudaStreamCreate(&stream);
	if (cudaSuccess != err) {
		LogError("cuda stream create failed\n");
		return false;
	}

	rt = createInferRuntime(logger);
	cue = rt->deserializeCudaEngine(blob.get(), size);
	ctx = cue->createExecutionContext();
	return true;
}

bool NNEngine::set_binding_shape(const char **id, size_t id_count, const Configurer &graph)
{
	for (size_t i = 0; i < id_count; ++i) {
		Json::Value val;
		if (!graph.get(id[i], val)) {
			LogError("get operator %s failed\n", id[i]);
			return false;
		}
		const Configurer opt(val);
		const std::string name = opt.get("name", std::string(""));
		if ("Input" != name && "Output" != name) continue;
		std::vector<int> shape = opt.get("shape", std::vector<int>());
		if (shape.size() < 1) {
			LogError("input shape for operator %s is invalid: %d\n", id[i], shape.size());
			return false;
		}
		std::string strs = "";
		for (auto d : shape) {
			strs = strs + std::to_string(d) + " ";
		}
		LogDebug("set binding shape: %s\n", strs.c_str());
		binding_shape.emplace_back(shape);
	}
	return true;
}

} // namespace tensorrt
} // namespace algorithm