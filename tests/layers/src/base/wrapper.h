/* A wrapper header: its portability #ifs do not count. */
#pragma once

#if defined(__APPLE__) || defined(__linux__)
#define FX_POSIX 1
#else
#define FX_POSIX 0
#endif
