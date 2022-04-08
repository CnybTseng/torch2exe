#include <cstring>

#include "utils/algorithmFactory.h"
#include "utils/deleter.h"

namespace algorithm {

// bool AlgorithmFactory::registry(const std::string &name, AlgorithmFactory::Creator creator)
// {
// 	auto it = get_creators().find(name);
// 	if (it == get_creators().end()) {
// 		get_creators()[name] = creator;
// 		return true;
// 	}
// 	return false;
// }
// 
// std::unique_ptr<Algorithm, Deleter> AlgorithmFactory::get_algorithm(const std::string &name)
// {
// 	auto it = get_creators().find(name);
// 	if (it != get_creators().end()) {
// 		return it->second(nullptr);
// 	}
// 	return std::unique_ptr<Algorithm, Deleter>(nullptr, destroyer);
// }
// 
// std::map<std::string, AlgorithmFactory::Creator> &AlgorithmFactory::get_creators(void)
// {
// 	static std::map<std::string, AlgorithmFactory::Creator> creators;
// 	return creators;
// }
// 
// AlgorithmListSP AlgorithmFactory::get_algorithm_list(void)
// {
// 	const int count = static_cast<int>(get_creators().size());
// 	char *data = new (std::nothrow) char[sizeof(AlgorithmList) + count * sizeof(Name)];
// 	if (!data) {
// 		LogError("allocate memory failed\n");
// 		return AlgorithmListSP(nullptr);
// 	}
// 	
// 	AlgorithmListSP algorithms = AlgorithmListSP(
// 		reinterpret_cast<AlgorithmList *>(data), ArrayDeleter("algorithm list"));
// 	algorithms->count = count;
// 	
// 	auto iter = get_creators().begin();
// 	for (size_t i = 0; iter != get_creators().end(); ++iter, ++i) {
// 		strcpy(algorithms->names[i], iter->first.c_str());
// 	}
// 	
// 	return algorithms;
// }

} // namespace algorithm