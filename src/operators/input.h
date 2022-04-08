#ifndef INPUT_H_
#define INPUT_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class Input : public Operator, public RegisteredInFactory<Operator, Input>
{
public:
	Input() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~Input() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // namespace tensorrt
} // namespace algorithm

#endif // INPUT_H_