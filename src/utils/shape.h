#ifndef SHAPE_H_
#define SHAPE_H_

#include <vector>

namespace algorithm {

inline int numel(const std::vector<int> &shape)
{
	int count = 1;
	for (auto d : shape) {
		count *= d;
	}
	return count;
}

} // namespace algorithm

#endif // SHAPE_H_