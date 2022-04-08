#ifndef NNPP_H_
#define NNPP_H_

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "operators/plugins/common.h"

namespace algorithm {

class NNPP
{
public:
	NNPP();
	virtual bool set_input_size(int inw, int inh);
	virtual bool set_static_param(int outw, int outh, bool align_center, bool kar, bool reverse_channel,
		const float *mean, const float *var_recip);
	virtual void forward(const uint8_t *d_in, float *d_out, cudaStream_t stream);
	virtual void backward_inplace(std::vector<Detection> &dets, const std::vector<int> &mask=std::vector<int>()) const;
	virtual ~NNPP();
private:
	void set_param();
	int inw;
	int inh;
	int outw;
	int outh;
	bool align_center;
	bool kar;
	bool reverse_channel;
	int padw;
	int padh;
	int w;
	int h;
	float scale;
	std::unique_ptr<float[]> h_mean;
	std::unique_ptr<float[]> h_var_recip;
	std::shared_ptr<float> d_mean;
	std::shared_ptr<float> d_var_recip;
};

} // namespace algorithm

#endif // NNPP_H_