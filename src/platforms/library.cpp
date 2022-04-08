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