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
 * @brief Path functional module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifdef _WIN32
#include <io.h> 
#define access _access_s
#else
#include <unistd.h>
#endif

#include "path.h"

namespace algorithm {

std::string filename_from_path(const std::string &path, bool extension)
{
#ifdef _WIN32
	char sep = '\\';
#else
	char sep = '/';
#endif
	size_t i = path.rfind(sep, path.length());
	size_t j = path.rfind(".", path.length());
	if (i != std::string::npos && j != std::string::npos) {
		return path.substr(i + 1, j - i - 1);
	}
	return "";
}

std::string replace_filename_suffix(const std::string &filename, const std::string &suffix)
{
	size_t i = filename.rfind(".", filename.length());
	if (i != std::string::npos) {
		return filename.substr(0, i + 1) + suffix;
	}
	return std::string("");
}

bool file_exists(const char *path)
{
	return access(path, 0) == 0;
}

}