#ifndef BATCHNORM2D_H_
#define BATCHNORM2D_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class BatchNorm2d : public Operator, public RegisteredInFactory<Operator, BatchNorm2d>
{
public:
	BatchNorm2d() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~BatchNorm2d() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // namespace tensorrt
} // namespace algorithm

#endif // BATCHNORM2D_H_