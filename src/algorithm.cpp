#include <cstdarg>

#include <cuda_runtime_api.h>

#include "algorithm.h"
#include "utils/algorithmFactory.h"
#include "utils/cuda.h"
#include "utils/deleter.h"
#include "utils/logger.h"

namespace algorithm {

std::shared_ptr<Image> Image::alloc(uint16_t h, uint16_t w, uint16_t s, bool pinned)
{
	std::shared_ptr<Image> image;
	const size_t size = h * s + sizeof(Image);
	if (pinned) {
		image.reset(reinterpret_cast<Image *>(cudaMallocHostWC(size)), cudaFreerHost("image"));
	} else {
		image.reset(reinterpret_cast<Image *>(new (std::nothrow)char[size]), ArrayDeleter("image"));
	}
	if (image) {
		image->width = w;
		image->height = h;
		image->stride = s;
	}
	return image;
}

std::unique_ptr<Algorithm, Deleter> Algorithm::create(const char *name)
{
	return AlgorithmFactory<Algorithm>::get_algorithm(name);
}

void Algorithm::register_logger(LogCallback fun)
{
	logging::register_logger(fun);
}

AlgorithmListSP Algorithm::get_algorithm_list()
{
	return AlgorithmFactory<Algorithm>::get_algorithm_list();
}

} // namespace algorithm