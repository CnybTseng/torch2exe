#ifndef DELETER_H_
#define DELETER_H_

#include <string>

#include "algorithm.h"
#include "utils/logger.h"

namespace algorithm {

struct BaseDeleter
{
	BaseDeleter(const std::string &signature_) : signature(signature_) {}
	std::string signature;
};

struct cudaFreer : public BaseDeleter
{
	cudaFreer(const std::string &signature_="") : BaseDeleter(signature_) {}
	void operator()(void *obj) const;
};

struct cudaFreerHost : public BaseDeleter
{
	cudaFreerHost(const std::string &signature_="") : BaseDeleter(signature_) {}
	void operator()(void *obj) const;
};

struct ArrayDeleter : public BaseDeleter
{
	ArrayDeleter(const std::string &signature_="") : BaseDeleter(signature_) {}
	void operator()(void *obj) const;
};

void cudafreer(void *p);

void cudafreerhost(void *p);

void array_deleter(void *p);

} // namespace algorithm

#endif // DELETER_H_