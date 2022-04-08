#ifndef YOLOV5_H_
#define YOLOV5_H_

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

class YOLOV5 : public Detector, public RegisteredInFactory<Algorithm, YOLOV5>
{
public:
	YOLOV5() = default;
	static std::unique_ptr<Algorithm, Deleter> create(const char *cfg);
	static std::string get_name(void);
	virtual ~YOLOV5() = default;
	virtual bool init(const char *cfg) override;
	virtual bool execute(const BlobSP &input, BlobSP &output) override;
	virtual void destroy(void) override;
private:
	NNPP nnpp;	
};

} // namespace detector
} // namespace algorithm

#endif // YOLOV5_H_