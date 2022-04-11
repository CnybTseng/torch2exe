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
 * @brief logging module.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#ifndef LOGGER_H_
#define LOGGER_H_

#include "algorithm.h"

/**
 * @brief Define default logging level macro.
 */
#if !defined(ALGLOG_TRACE) && !defined(ALGLOG_DEBUG) && !defined(ALGLOG_INFO) && !defined(ALGLOG_WARN) && !defined(ALGLOG_ERROR) && !defined(ALGLOG_FATAL)
#define ALGLOG_TRACE
#endif

#ifdef ALGLOG_TRACE
#define LogTrace(format, ...) algorithm::logging::_log("%s %s TRACE %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogDebug(format, ...) algorithm::logging::_log("%s %s DEBUG %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogInfo(format, ...) algorithm::logging::_log("%s %s INFO %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogWarn(format, ...) algorithm::logging::_log("%s %s WARN %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogError(format, ...) algorithm::logging::_log("%s %s ERROR %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogFatal(format, ...) algorithm::logging::_log("%s %s FATAL %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)

#elif defined(ALGLOG_DEBUG)
#define LogTrace(format, ...)
#define LogDebug(format, ...) algorithm::logging::_log("%s %s DEBUG %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogInfo(format, ...) algorithm::logging::_log("%s %s INFO %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogWarn(format, ...) algorithm::logging::_log("%s %s WARN %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogError(format, ...) algorithm::logging::_log("%s %s ERROR %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogFatal(format, ...) algorithm::logging::_log("%s %s FATAL %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)

#elif defined(ALGLOG_INFO)
#define LogTrace(format, ...)
#define LogDebug(format, ...)
#define LogInfo(format, ...) algorithm::logging::_log("%s %s INFO %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogWarn(format, ...) algorithm::logging::_log("%s %s WARN %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogError(format, ...) algorithm::logging::_log("%s %s ERROR %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogFatal(format, ...) algorithm::logging::_log("%s %s FATAL %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)

#elif defined(ALGLOG_WARN)
#define LogTrace(format, ...)
#define LogDebug(format, ...)
#define LogInfo(format, ...)
#define LogWarn(format, ...) algorithm::logging::_log("%s %s WARN %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogError(format, ...) algorithm::logging::_log("%s %s ERROR %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogFatal(format, ...) algorithm::logging::_log("%s %s FATAL %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)

#elif defined(ALGLOG_ERROR)
#define LogTrace(format, ...)
#define LogDebug(format, ...)
#define LogInfo(format, ...)
#define LogWarn(format, ...)
#define LogError(format, ...) algorithm::logging::_log("%s %s ERROR %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LogFatal(format, ...) algorithm::logging::_log("%s %s FATAL %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)

#elif defined(ALGLOG_FATAL)
#define LogTrace(format, ...)
#define LogDebug(format, ...)
#define LogInfo(format, ...)
#define LogWarn(format, ...)
#define LogError(format, ...)
#define LogFatal(format, ...) algorithm::logging::_log("%s %s FATAL %s:%d " format, __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__)

#else
#define LogTrace(format, ...)
#define LogDebug(format, ...)
#define LogInfo(format, ...)
#define LogWarn(format, ...)
#define LogError(format, ...)
#define LogFatal(format, ...)
#endif

namespace algorithm {
namespace logging {

/**
 * @brief Register logging function.
 * @param fun Logging callback function.
 */
void register_logger(LogCallback fun);

/**
 * @brief Log messages.
 * @warning DO NOT CALL THIS FUNCTION DIRECTLY! Use macro instead.
 * @param format C string that contains a format string that follows the same specifications as format in printf.
 */
void _log(const char *format, ...);

} // namespace logging
} // namespace algorithm

#endif // LOGGER_H_