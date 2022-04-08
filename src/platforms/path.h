#ifndef PATH_H_
#define PATH_H_

#include <string>

namespace algorithm {

std::string filename_from_path(const std::string &path, bool extension=false);
std::string replace_filename_suffix(const std::string &filename, const std::string &suffix);
bool file_exists(const char *path);

} // namespace algorithm

#endif // PATH_H_