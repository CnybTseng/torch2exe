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