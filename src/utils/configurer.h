#ifndef CONFIGURER
#define CONFIGURER

#include <memory>
#include <string>
#include <vector>

#include <json/json.h>

namespace algorithm {

class Configurer
{
public:
	Configurer() = default;
	Configurer(const Json::Value &val) : root(val) {}
	virtual ~Configurer() = default;
	bool init(const char *cfg);
	bool init(const char *data, size_t len);
	const Json::Value &get() const {return root;};
	std::string get(const char *key, const std::string &default_val) const;
	bool get(const char *key, const bool &default_val) const;
	int get(const char *key, const int &default_val) const;
	float get(const char *key, const float &default_val) const;
	double get(const char *key, const double &default_val) const;
	std::vector<std::string> get(const char *key, const std::vector<std::string> &default_val) const;
	std::vector<int> get(const char *key, const std::vector<int> &default_val) const;
	std::vector<float> get(const char *key, const std::vector<float> &default_val) const;
	bool get(const char *key, Json::Value &val) const;
private:
	std::shared_ptr<char[]> buffer;
	Json::Value root;
};

} // namespace algorithm

#endif // CONFIGURER