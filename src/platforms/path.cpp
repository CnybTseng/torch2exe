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