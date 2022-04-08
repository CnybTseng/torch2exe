#ifndef CONV2D_H_
#define CONV2D_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class Conv2d : public Operator, public RegisteredInFactory<Operator, Conv2d>
{
public:
	Conv2d() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~Conv2d() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // tensorrt
} // namespace algorithm

#endif // CONV2D_H_