#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "resize.h"

namespace algorithm {

using DTYPE = float;

static const int CHANNELS = 4;

__global__ void bilinear_interpolate(cudaTextureObject_t text_obj, DTYPE *out, int width, int height, float scale)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float tu = scale * (x + 0.5) - 0.5;
	float tv = scale * (y + 0.5) - 0.5;
	auto val = tex2D<float4>(text_obj, tu, tv);
	out[y * width * CHANNELS + x * CHANNELS] = val.x;
	out[y * width * CHANNELS + x * CHANNELS + 1] = val.y;
	out[y * width * CHANNELS + x * CHANNELS + 2] = val.z;
	out[y * width * CHANNELS + x * CHANNELS + 3] = val.w;
}

void resize(void)
{
	printf("resize ...\n");
	static const int width = 1920;
	static const int height = 1080;
	
	cv::Mat bgr = cv::imread("test.jpg");
	cv::Mat bgra;
	cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
	
	cv::imwrite("bgra.png", bgra);
	
	cv::Mat bgra_float;
	bgra.convertTo(bgra_float, CV_32FC4);
	
	DTYPE *data = (DTYPE *)malloc(width * height * CHANNELS * sizeof(DTYPE));
	memcpy(data, bgra_float.data, width * height * CHANNELS * sizeof(DTYPE));
	
	cudaChannelFormatDesc chan_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cudaArray_t array;
	cudaError_t err = cudaMallocArray(&array, &chan_desc, width, height);
	if (err != cudaSuccess) {
		printf("cudaMallocArray failed\n");
		return;
	}
	
	static const int out_width = 1280;
	static const int out_height = 704;
	static const float scale = width / (float)out_width;
	
	DTYPE *d_out;
	cudaMalloc(&d_out, out_width * out_height * CHANNELS * sizeof(DTYPE));
	
	DTYPE *h_out = (DTYPE *)malloc(out_width * out_height * CHANNELS * sizeof(DTYPE));

	const size_t spitch = width * CHANNELS * sizeof(DTYPE);
	static const size_t wOffset = 0;
	static const size_t hOffset = 0;

	cudaMemcpy2DToArray(array, wOffset, hOffset, data, spitch, spitch, height, cudaMemcpyHostToDevice);

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = array;

	cudaTextureDesc text_desc;
	memset(&text_desc, 0, sizeof(text_desc));
	text_desc.addressMode[0] = cudaAddressModeClamp;
	text_desc.addressMode[1] = cudaAddressModeClamp;
	text_desc.filterMode = cudaFilterModeLinear;
	text_desc.readMode = cudaReadModeElementType; // cudaReadModeNormalizedFloat
	text_desc.normalizedCoords = 0;
	
	cudaTextureObject_t text_obj = 0;
	cudaCreateTextureObject(&text_obj, &res_desc, &text_desc, nullptr);
	
	dim3 block(16, 16);
	dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
	bilinear_interpolate<<<grid, block>>>(text_obj, d_out, out_width, out_height, scale);
	
	cudaMemcpy(h_out, d_out, out_width * out_height * CHANNELS * sizeof(DTYPE), cudaMemcpyDeviceToHost);
	
	cv::Mat resized_float32(out_height, out_width, CV_32FC4);
	memcpy(resized_float32.data, h_out, out_width * out_height * CHANNELS * sizeof(DTYPE));
	
	cv::Mat resized_uint8;
	resized_float32.convertTo(resized_uint8, CV_8UC4);
	
	cv::Mat resized;
	cv::cvtColor(resized_uint8, resized, cv::COLOR_BGRA2BGR);
	cv::imwrite("resized.png", resized);
	
	cudaDestroyTextureObject(text_obj);
	cudaFreeArray(array);
	cudaFree(d_out);
	free(h_out);
	free(data);
	printf("resize done.\n");
}

} // algorithm