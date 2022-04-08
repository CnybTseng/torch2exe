#ifndef ALGORITHM_FACTORY_H_
#define ALGORITHM_FACTORY_H_

#ifdef __GNUC__
#include <cxxabi.h>
#endif // __GNUC__

#include <map>
#include <memory>
#include <regex>
#include <string>

#include "algorithm.h"
#include "utils/logger.h"

namespace algorithm {

template <typename B>
class AlgorithmFactory
{
	using Creator = std::unique_ptr<B, Deleter>(*)(const char *);
public:
	AlgorithmFactory() = delete;
	virtual ~AlgorithmFactory() = default;
	static bool registry(const std::string &name, Creator creator)
	{
		auto it = get_creators().find(name);
		if (it == get_creators().end()) {
			get_creators()[name] = creator;
			return true;
		}
		return false;
	}
	
	static std::unique_ptr<B, Deleter> get_algorithm(const std::string &name)
	{
		auto it = get_creators().find(name);
		if (it != get_creators().end()) {
			return it->second(nullptr);
		}
		return std::unique_ptr<B, Deleter>(nullptr, [](void *p){});
	}
	
	static AlgorithmListSP get_algorithm_list(void)
	{
		const int count = static_cast<int>(get_creators().size());
		char *data = new (std::nothrow) char[sizeof(AlgorithmList) + count * sizeof(Name)];
		if (!data) {
			LogError("allocate memory failed\n");
			return AlgorithmListSP(nullptr);
		}
		
		AlgorithmListSP algorithms = AlgorithmListSP(
			reinterpret_cast<AlgorithmList *>(data), [](void *p){
				LogDebug("delete [] algorithm list\n");
				delete [] (char *)p;
			}
		);
		algorithms->count = count;
		
		auto iter = get_creators().begin();
		for (size_t i = 0; iter != get_creators().end(); ++iter, ++i) {
			strcpy(algorithms->names[i], iter->first.c_str());
		}
		
		return algorithms;
	}
private:
	static std::map<std::string, Creator> &get_creators(void)
	{
		static std::map<std::string, AlgorithmFactory::Creator> creators;
		return creators;
	}
};

template <typename B, typename T>
class RegisteredInFactory
{
public:
	RegisteredInFactory() {(void)registered;}
	virtual ~RegisteredInFactory() = default;
private:
	static std::string demangle(const char *name)
	{
#ifdef _MSC_VER
		return name;
#elif defined(__GNUC__)
		int status = -4;
		std::unique_ptr<char, void (*)(void *)> res {
			abi::__cxa_demangle(name, nullptr, nullptr, &status),
			free
		};
		return (status == 0) ? res.get() : name;
#else
#error unexpected C compiler (MSC/GCC)
#endif
	}
	static bool registry(void)
	{
		const auto &name = demangle(typeid(T).name());
		static const std::string key("::");
		size_t found = name.rfind(key);
		if (found != std::string::npos) {
			const std::string cname = name.substr(found + key.size());
			LogDebug("register %s in factory\n", name.c_str());
			return AlgorithmFactory<B>::registry(cname, T::create);
		}
		LogWarn("use manuall named algorithm name instead\n");
		return AlgorithmFactory<B>::registry(T::get_name(), T::create);
	}
private:
	static bool registered;
};

template <typename B, typename T>
bool RegisteredInFactory<B, T>::registered = RegisteredInFactory<B, T>::registry();

} // namespace algorithm

#endif // ALGORITHM_FACTORY_H_