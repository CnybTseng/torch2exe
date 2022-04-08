#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <vector>
#include <memory>

#include <cuda_runtime_api.h>

#include "algorithm.h"
#include "algorithmExt.h"
#include "nnEngine.h"
#include "operators/nnpp.h"

namespace algorithm {

class Detector : public AlgorithmExt
{
public:
	Detector();
	virtual ~Detector() = default;
protected:
	virtual bool parse_config(const char *cfg) override;
	virtual bool init(const char *cfg) override;
	virtual bool preprocess(const BlobSP &input, NNPP &nnpp, float pad_val);
	virtual bool postprocess(const NNPP &nnpp, BlobSP &output);
	virtual void destroy(void) override;
	int device;
	int batch_size;
	float nms_thresh;
	float score_thresh;
	std::vector<std::string> categories;
	tensorrt::NNEngine nne;
	std::vector<int> shape_in;
	std::vector<int> shape_out;
	std::shared_ptr<uint8_t> d_img;
	std::shared_ptr<float> d_in;
	std::shared_ptr<float> d_out;
	std::unique_ptr<float[]> h_in;
	std::unique_ptr<float[]> h_out;
private:
	virtual void reset_num_objs();
};

} // namespace algorithm

#endif // DETECTOR_H_