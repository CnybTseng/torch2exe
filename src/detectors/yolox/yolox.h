#ifndef YOLOX_H_
#define YOLOX_H_

#include <memory>
#include <string>
#include <vector>

#include "algorithm.h"
#include "detector.h"
#include "operators/nnpp.h"
#include "nnEngine.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace detector {

class YOLOX : public Detector, public RegisteredInFactory<Algorithm, YOLOX>
{
public:
	YOLOX() = default;
	static std::unique_ptr<Algorithm, Deleter> create(const char *cfg);
	static std::string get_name(void);
	virtual ~YOLOX() = default;
	virtual bool init(const char *cfg) override;
	virtual bool execute(const BlobSP &input, BlobSP &output) override;
	virtual void destroy(void) override;
private:
	NNPP nnpp;
};

} // namespace detector
} // namespace algorithm

#endif // YOLOX_H_