#ifndef MAXPOOL2D_H_
#define MAXPOOL2D_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class MaxPool2d : public Operator, public RegisteredInFactory<Operator, MaxPool2d>
{
public:
	MaxPool2d() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~MaxPool2d() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // namespace tensorrt
} // namespace algorithm

#endif // MAXPOOL2D_H_