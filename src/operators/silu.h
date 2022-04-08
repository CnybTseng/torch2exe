#ifndef SILU_H_
#define SILU_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class SiLU : public Operator, public RegisteredInFactory<Operator, SiLU>
{
public:
	SiLU() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~SiLU() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // namespace tensorrt
} // namespace algorithm

#endif // SILU_H_