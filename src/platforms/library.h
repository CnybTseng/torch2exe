#ifndef LIBRARY_H_
#define LIBRARY_H_

namespace algorithm {

void *open_dll(const char *path);
void *get_dlsym(const void *handle, const char *name);
void close_dll(void *handle);

} // namespace algorithm

#endif