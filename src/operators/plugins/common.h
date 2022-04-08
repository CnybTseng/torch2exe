#ifndef COMMON_H_
#define COMMON_H_

namespace algorithm {

struct alignas(float) Box
{
	float x;			///! box center x
	float y;			///! box center y
	float width;		///! box width
	float height;		///! box height
};

struct alignas(float) Detection
{
	Box box;			///! bounding box
	float score;		///! product of objectness and class probability
	float category;		///! class identifier
};

template <typename T>
void write(char*& buffer, const T& val)
{
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

template <typename T>
T read(const char*& buffer)
{
	T val = *reinterpret_cast<const T*>(buffer);
	buffer += sizeof(T);
	return val;
}

} // namespace algorithm

#endif // COMMON_H_