#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <json/json.h>
#include <NvInfer.h>

#include "algorithm.h"

namespace algorithm {
namespace tensorrt {

/**
 * @struct NetworkContext
 * @brief 神经网络定义上线文.
 */
struct NetworkContext
{
	float *weight;														///! 模型权重
	size_t len_read;													///! 模型权重已读量
	std::vector<std::shared_ptr<float[]>> derived_weights;				///! 临时参数
	std::vector<std::shared_ptr<nvinfer1::IPluginV2>> plugins;			///! 临时插件
	std::shared_ptr<nvinfer1::INetworkDefinition> network;				///! 神经网络定义
	std::map<std::string, std::vector<nvinfer1::ITensor *>> output;		///! 算子输出
};

/**
 * @class Operator
 * @brief 高级用户使用的算子接口类.
 * @details 高级用户必须重载Operater类中定义的抽象方法.
 *  例如, 实现Conv2d算子:

 * #include "operator.h"
 * #include "utils/algorithmFactory.h"
 * 
 * class Conv2d : public Operator, public RegisteredInFactory<Operator, Conv2d>
 * {
 * public:
 * 	  Conv2d() = default;
 * 	  static std::unique_ptr<Operator, Deleter> create(const char *name);
 * 	  static std::string get_name(void);
 * 	  virtual ~Conv2d() = default;
 * 	  virtual bool set(const Json::Value &cfg, nvinfer1::INetworkDefinition *&network) override;
 * 	  virtual void destroy(void) override;
 * };
 *
 * @note 想要将算子注册到算子库里, 必须做两件事:
 *  1. 继承`RegisteredInFactory`类.
 *  2. 定义和实现方法`static std::unique_ptr<Operator, Deleter> create(const char *cfg)`.
 *
 * @warning 如果没有定义和实现`create`方法, 程序编译及运行时不会报错,
 *  但您的算子没有注册到算子库里, 将不可用!
 *
 */
class Operator
{
public:
	Operator() = default;
	virtual ~Operator() = default;
	
	/**
	 * @brief 创建算子实例.
	 * @param cfg 算子配置, 为JSON格式字符串.
	 * @param network 神经网络定义.
	 * @return 算子设置状态. 成功: true, 失败: false.
	 */
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) = 0;
	
	/**
	 * @brief 销毁算子实例.
	 */
	virtual void destroy(void) = 0;
};

} // namespace tensorrt
} // namespace algorithm

#endif // OPERATOR_H_