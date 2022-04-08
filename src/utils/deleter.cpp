#include <cuda_runtime_api.h>

#include "utils/deleter.h"
#include "utils/logger.h"

namespace algorithm {

void cudaFreer::operator()(void *obj) const
{
	LogDebug("cudaFree %s\n", signature.c_str());
	cudaFree(obj);
}

void cudaFreerHost::operator()(void *obj) const
{
	LogDebug("cudaFreeHost %s\n", signature.c_str());
	if (obj) {
		cudaFreeHost(obj);
	}
}

void ArrayDeleter::operator()(void *obj) const
{
	LogDebug("delete [] %s\n", signature.c_str());
	if (obj) {
		delete [] obj;
	}
}

void cudafreer(void *p)
{
	LogDebug("cudaFree\n");
	cudaFree(p);
}

void cudafreerhost(void *p)
{
	LogDebug("cudaFreeHost\n");
	if (p) {
		cudaFreeHost(p);
	}
}

void array_deleter(void *p)
{
	LogDebug("delete []\n");
	if (p) {
		delete [] p;
	}
}

} // namespace algorithm