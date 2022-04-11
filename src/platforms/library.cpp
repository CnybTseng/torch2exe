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
 * @brief Dynamic library load module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifdef _WIN32
#include <windows.h>
#else
#include<dlfcn.h>
#endif

#include "library.h"
#include "utils/logger.h"

namespace algorithm {

void *open_dll(const char *path)
{
#ifdef _WIN32
	HINSTANCE handle = LoadLibrary(path);
	return (void *)handle;
#else
	void *handle = dlopen(path, RTLD_LAZY);
	if (handle) {
		dlerror();
	}
	return handle;
#endif	
}

void *get_dlsym(const void *handle, const char *name)
{
#ifdef _WIN32
	HINSTANCE handle_ = (HINSTANCE)handle;
	void *sym = GetProcAddress(handle_, name);
	if (!sym) {
		LogDebug("GetProcAddress failed: %d\n", GetLastError());
	}
	return sym;
#else
	void *sym = dlsym(const_cast<void *>(handle), name);
	char *error = dlerror();
	if (error) {
		LogDebug("dlsym failed: %s\n", error);
		return nullptr;
	}
	return sym;
#endif
}

void close_dll(void *handle)
{
#ifdef _WIN32
	HINSTANCE handle_ = (HINSTANCE)handle;
	FreeLibrary(handle_);
#else
	dlclose(handle);
#endif	
}

} // namespace algorithm