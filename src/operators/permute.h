#ifndef PERMUTE_H_
#define PERMUTE_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class Permute : public Operator, public RegisteredInFactory<Operator, Permute>
{
public:
	Permute() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~Permute() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // namespace tensorrt
} // namespace algorithm

#endif // PERMUTE_H_