#include <cstring>

#include <cuda_runtime_api.h>

#include "utils/cuda.h"
#include "utils/logger.h"

namespace algorithm {

void *cudaMallocWC(size_t size)
{
	void *ptr = nullptr;
	cudaError_t err = cudaMalloc(&ptr, size);
	if (cudaSuccess != err) {
		LogError("cuda malloc host failed\n");
		return nullptr;
	}
	return ptr;
}

void *cudaMallocHostWC(size_t size)
{
	void *ptr = nullptr;
	cudaError_t err = cudaMallocHost(&ptr, size);
	if (cudaSuccess != err) {
		LogError("cuda malloc host failed\n");
		return nullptr;
	}
	return ptr;
}

std::string cudaDeviceName(int device)
{
	cudaDeviceProp prop;
	cudaError_t err = cudaGetDeviceProperties(&prop, device);
	if (err != cudaSuccess) {
		LogError("cudaGetDeviceProperties failed\n");
		return "";
	}
	
	for (size_t i = 0; i < strlen(prop.name); ++i) {
		if (32 == prop.name[i]) {
			prop.name[i] = '-';
		} else if (prop.name[i] > 64 && prop.name[i] < 91) {
			prop.name[i] += 32;
		}
	}
	
	return std::string(prop.name);
}

}