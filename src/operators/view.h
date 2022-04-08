#ifndef VIEW_H_
#define VIEW_H_

#include <json/json.h>

#include "algorithm.h"
#include "operator.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace tensorrt {

class View : public Operator, public RegisteredInFactory<Operator, View>
{
public:
	View() = default;
	static std::unique_ptr<Operator, Deleter> create(const char *name);
	static std::string get_name(void);
	virtual ~View() = default;
	virtual bool set(const char *id, const Json::Value &cfg, NetworkContext &ctx) override;
	virtual void destroy(void) override {}
};

} // namespace tensorrt
} // namespace algorithm

#endif // VIEW_H_