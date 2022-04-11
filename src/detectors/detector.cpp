/******************************************************************
 * PyTorch to executable program (torch2exe).
 * Copyright © 2022 Zhiwei Zeng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the “Software”), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 *
 * This file is part of torch2exe.
 *
 * @file
 * @brief Object detection.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cstring>

#include <cuda.h>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "detectors/nms.h"
#include "utils/configurer.h"
#include "utils/cuda.h"
#include "utils/logger.h"
#include "utils/deleter.h"
#include "utils/shape.h"

namespace algorithm {

Detector::Detector() :
	d_img(nullptr, cudaFreer("d_img")),
	d_in(nullptr, cudaFreer("d_in")),
	d_out(nullptr, cudaFreer("d_out"))
{
}

bool Detector::parse_config(const char *cfg)
{
	Configurer cfgr;
	if (!cfgr.init(cfg)) return false;

	device = cfgr.get("device", device);
	LogDebug("device: %d\n", device);

	batch_size = cfgr.get("batch_size", int(1));
	LogDebug("batch_size: %d\n", batch_size);

	nms_thresh = cfgr.get("nms_thresh", nms_thresh);
	LogDebug("nms_thresh: %f\n", nms_thresh);

	score_thresh = cfgr.get("score_thresh", score_thresh);
	LogDebug("score_thresh: %f\n", score_thresh);
		
	categories = cfgr.get("categories", categories);
	for (auto e : categories) {
		LogDebug("we will detect %s\n", e.c_str());
	}

	return true;
}

bool Detector::init(const char *cfg)
{
	cudaSetDevice(device);
	if (!parse_config(cfg)) {
		return false;
	}

	if (!nne.build(cfg)) {
		return false;
	}
	
	shape_in = nne.get_binding_shape(0);
	if (shape_in.empty()) return false;
	shape_out = nne.get_binding_shape(1);
	if (shape_out.empty()) return false;

	d_img.reset(reinterpret_cast<uint8_t *>(cudaMallocWC(1920 * 1080 * 3)));
	if (!d_img) return false;

	d_in.reset(reinterpret_cast<float *>(cudaMallocWC(numel(shape_in) * sizeof(float))));
	if (!d_in) return false;

	d_out.reset(reinterpret_cast<float *>(cudaMallocWC(numel(shape_out) * sizeof(float))));
	if (!d_out) return false;

	h_in.reset(new (std::nothrow) float[numel(shape_in)]);
	if (!h_in) {
		LogError("allocate memory failed\n");
		return false;
	}

	h_out.reset(new (std::nothrow) float[numel(shape_out)]);
	if (!h_out) {
		LogError("allocate memory failed\n");
		return false;
	}
	
	return true;
}

bool Detector::preprocess(const BlobSP &input, NNPP &nnpp, float pad_val)
{
	const ImageSP img = std::static_pointer_cast<Image>(input);
	if (!img) {
		LogError("nullptr to input blob\n");
		return false;
	}
	
	LogDebug("copy data to device memory\n");
	size_t size = img->stride * img->height;
	const cudaStream_t &stream = nne.get_stream();
	cudaError_t err = cudaMemcpyAsync(d_img.get(), img->data, size, cudaMemcpyHostToDevice, stream);
	if (cudaSuccess != err) {
		LogError("copy data to device failed\n");
		return false;
	}

	if (nnpp.set_input_size(img->width, img->height)) {
		volatile union
		{
			float f;
			int i;
		} pad_val_u;
		pad_val_u.f = pad_val;
		cuMemsetD32Async((CUdeviceptr)(d_in.get()), pad_val_u.i, numel(shape_in), stream);
	}
	nnpp.forward(d_img.get(), d_in.get(), stream);
	// {
	// 	cudaMemcpyAsync(h_in.get(), d_in.get(), numel(shape_in) * sizeof(float), cudaMemcpyDeviceToHost, stream);
	// 	cudaStreamSynchronize(stream);
	// 	cv::Mat bluef(shape_in[2], shape_in[3], CV_32FC1, h_in.get());
	// 	cv::Mat blue;
	// 	bluef.convertTo(blue, CV_8UC1);
	// 	cv::imwrite("blue.png", blue);
	// 	cv::Mat bgr(img->height, img->width, CV_8UC3, img->data);
	// 	cv::imwrite("bgr.png", bgr);
	// }
	reset_num_objs();
	return true;
}

bool Detector::postprocess(const NNPP &nnpp, BlobSP &output)
{
	const cudaStream_t &stream = nne.get_stream();
	cudaMemcpyAsync(h_out.get(), d_out.get(), numel(shape_out) * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	size_t count = static_cast<size_t>(((int *)h_out.get())[0]);
	LogDebug("count = %d\n", count);
	if (count > 1000) {
		LogWarn("too many objects\n");
		count = 1000;
	}
	if (0 == count) {
		return true;
	}
	Detection *first = reinterpret_cast<Detection *>(&h_out.get()[1]);
	std::vector<Detection> detections(first, first + count);

	LogTrace("nms\n");
	std::vector<int> picked;
	nms(detections, picked, score_thresh, nms_thresh);
	if (0 == picked.size()) {
		return true;
	}
	nnpp.backward_inplace(detections, picked);
	LogTrace("nms done\n");

	LogTrace("allocate output buffer\n");
	size_t size = sizeof(DetectorOutput) + picked.size() * sizeof(DetectorOutputData);
	output.reset(reinterpret_cast<DetectorOutput *>(new (std::nothrow) char[size]), ArrayDeleter("detector output"));
	if (!output) {
		LogError("allocate memory failed\n");
		return false;
	}
	LogTrace("allocate output buffer done\n");
	
	LogTrace("fill output buffer %d %d\n", picked.size(), detections.size());
	DetectorOutputSP obj = std::static_pointer_cast<DetectorOutput>(output);
	obj->type = DETECTION;
	obj->count = static_cast<uint16_t>(picked.size());
	for (int i = 0; i < obj->count; ++i) {
		const Detection &det = detections[picked[i]];
		int category = static_cast<int>(det.category);
		strcpy(obj->data[i].category, categories[category].c_str());
		float x = (det.box.x - det.box.width * .5f);
		float y = (det.box.y - det.box.height * .5f);
		obj->data[i].box.x = static_cast<uint16_t>(x);
		obj->data[i].box.y = static_cast<uint16_t>(y);
		obj->data[i].box.width = static_cast<uint16_t>(det.box.width);
		obj->data[i].box.height = static_cast<uint16_t>(det.box.height);
		obj->data[i].score = det.score;
	}
	LogTrace("fill output buffer done\n");
	return true;
}

void Detector::destroy(void)
{
	LogInfo("Detector::destroy\n");
	nne.destroy();
}

void Detector::reset_num_objs()
{
	const cudaStream_t &stream = nne.get_stream();
	int batch_stride = 1;
	for (size_t i = 1; i < shape_out.size(); ++i) {
		batch_stride *= shape_out[i];
	}
	for (int b = 0; b < shape_out[0]; ++b) {
		cudaMemsetAsync(d_out.get() + b * batch_stride, 0, sizeof(float), stream);
	}
}

} // namespace algorithm