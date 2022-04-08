#include <cstdarg>

#include "logger.h"

namespace algorithm {
namespace logging {

static void default_logger(const char *format, va_list arg)
{	
	vfprintf(stdout, format, arg);
}

static LogCallback logger = default_logger;

void register_logger(LogCallback fun)
{
	logger = fun;
}

void _log(const char *format, ...)
{
	va_list ap;
	va_start(ap, format);
	logger(format, ap);
	va_end(ap);
}

} // namespace logging
} // namespace algorithm