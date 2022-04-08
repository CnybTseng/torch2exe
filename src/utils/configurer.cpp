#include <fstream>
#include <memory>

#include "configurer.h"
#include "utils/logger.h"
#include "utils/deleter.h"

namespace algorithm {

bool Configurer::init(const char *cfg)
{
	std::ifstream ifs(cfg);
	if (!ifs.good()) {
		LogError("read configuration file failed: %s\n", cfg);
		return false;
	}
	
	ifs.seekg(0, ifs.end);
	const size_t size = ifs.tellg();
	ifs.seekg(0, ifs.beg);

	buffer.reset(new (std::nothrow) char[size], [](char *p){
		LogDebug("delete [] configuration buffer\n");
		delete [] p;
	});
	if (!buffer) {
		ifs.close();
		LogError("allocate memory failed\n");
		return false;
	}
	
	ifs.read(buffer.get(), size);
	ifs.close();

	JSONCPP_STRING err;
    Json::CharReaderBuilder builder;
	const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
	bool ret = reader->parse(buffer.get(), buffer.get() + size, &root, &err);
	if (!ret) {
		LogError("parse json string failed\n");
		return false;
	}

	return true;
}

bool Configurer::init(const char *data, size_t len)
{
	JSONCPP_STRING err;
    Json::CharReaderBuilder builder;
	const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
	bool ret = reader->parse(data, data + len, &root, &err);
	if (!ret) {
		LogError("parse json string failed\n");
		return false;
	}

	return true;
}

std::string Configurer::get(const char *key, const std::string &default_val) const
{
	if (root.isMember(key)) {
		return root[key].asString();
	}
	return default_val;
}

bool Configurer::get(const char *key, const bool &default_val) const
{
	if (root.isMember(key)) {
		return root[key].asBool();
	}
	return default_val;
}

int Configurer::get(const char *key, const int &default_val) const
{
	if (root.isMember(key)) {
		return root[key].asInt();
	}
	return default_val;
}

float Configurer::get(const char *key, const float &default_val) const
{
	if (root.isMember(key)) {
		return root[key].asFloat();
	}
	return default_val;
}

double Configurer::get(const char *key, const double &default_val) const
{
	if (root.isMember(key)) {
		return root[key].asDouble();
	}
	return default_val;
}

std::vector<std::string> Configurer::get(const char *key, const std::vector<std::string> &default_val) const
{
	if (root.isMember(key)) {
		std::vector<std::string> arr;
		Json::Value::const_iterator iter;
		Json::Value val = root[key];
		for (iter = val.begin(); iter != val.end(); iter++) {
			arr.emplace_back(iter->asString());
		}
		return arr;
	}
	return default_val;
}

std::vector<int> Configurer::get(const char *key, const std::vector<int> &default_val) const
{
	if (root.isMember(key)) {
		std::vector<int> arr;
		Json::Value::const_iterator iter;
		Json::Value val = root[key];
		for (iter = val.begin(); iter != val.end(); iter++) {
			arr.emplace_back(iter->asInt());
		}
		return arr;
	}
	return default_val;
}

std::vector<float> Configurer::get(const char *key, const std::vector<float> &default_val) const
{
	if (root.isMember(key)) {
		std::vector<float> arr;
		Json::Value::const_iterator iter;
		Json::Value val = root[key];
		for (iter = val.begin(); iter != val.end(); iter++) {
			arr.emplace_back(iter->asFloat());
		}
		return arr;
	}
	return default_val;
}

bool Configurer::get(const char *key, Json::Value &val) const
{
	if (root.isMember(key)) {
		val = root[key];
		return true;
	}
	return false;
}

} // namespace algorithm