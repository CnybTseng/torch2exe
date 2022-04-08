#ifndef ALGORITHM_EXT_H_
#define ALGORITHM_EXT_H_

#include "algorithm.h"

namespace algorithm {

/**
 * @class AlgorithmExt
 * @brief 高级用户使用的算法接口类.
 * @details 高级用户必须要实现`Algorithm`类和`AlgorithmExt`类中定义的纯虚函数.
 *  例如, 实现FasterRCNN算法:
 *
 *  #include "algorithm.h"
 *  #include "algorithmExt.h"
 *  #include "utils/algorithmFactory.h"
 *  
 *  class FasterRCNN : public AlgorithmExt, public RegisteredInFactory<FasterRCNN>
 *  {
 *  public:
 *  	FasterRCNN() = default;
 *  	static std::unique_ptr<Algorithm, Deleter> create(const char *cfg);
 *  	virtual ~FasterRCNN() = default;
 *  	virtual bool init(const char *cfg) override;
 *  	virtual bool execute(const BlobSP &input, BlobSP &output) override;
 *  	virtual void destroy(void) override;
 *  private:
 *  	virtual bool parse_config(const char *cfg) override;
 *  };
 *
 * @note 想要将算法注册到算法库里, 必须做两件事:
 *  1. 继承`RegisteredInFactory`类.
 *  2. 定义和实现方法`static std::unique_ptr<Algorithm, Deleter> create(const char *cfg)`.
 *
 * @warning 如果没有定义和实现`create`方法, 程序编译及运行时不会报错,
 *  但您的算法没有注册到算法库里, 将不可用!
 *
 */
class AlgorithmExt : public Algorithm
{
public:
	AlgorithmExt() = default;
	virtual ~AlgorithmExt() = default;
	
	/**
	 * @brief 销毁算法实例.
	 */
	virtual void destroy(void) = 0;
private:
	/**
	 * @brief 解析配置文件.
	 * @param cfg 算法配置, 为JSON格式字符串.
	 * @return 配置文件解析状态. 成功: true, 失败: false.
	 */
	virtual bool parse_config(const char *cfg) = 0;
};

} // namespace algorithm

#endif // ALGORITHM_EXT_H_