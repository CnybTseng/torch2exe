/******************************************************************
 * PyTorch to executable program (torch2exe).
 * Copyright © 2022 Zhiwei Zeng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the “Software”), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 *
 * This file is part of torch2exe.
 *
 * @file
 * @brief Configuration functional module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

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
		if (p) {
			LogDebug("delete [] configuration buffer\n");
			delete [] p;
		}
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