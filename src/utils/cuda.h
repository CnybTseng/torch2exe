#ifndef CUDA_H_
#define CUDA_H_

#include <string>

namespace algorithm {

void *cudaMallocWC(size_t size);
void *cudaMallocHostWC(size_t size);
std::string cudaDeviceName(int device);

} // namespace algorithm

#endif // CUDA_H_