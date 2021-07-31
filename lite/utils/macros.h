// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(class__) \
  class__(const class__&) = delete;       \
  class__& operator=(const class__&) = delete;
#endif

#define LITE_UNIMPLEMENTED CHECK(false) << "Not Implemented";

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

/*
#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif

#ifndef UNLIKELY
//#define UNLIKELY(x) __built_expect(!!(x), 0)
#define UNLIKELY(x) (x)
#endif
 */

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif

#if defined(__FLT_MAX__)
#define FLT_MAX __FLT_MAX__
#endif  // __FLT_MAX__

#if (defined __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && \
    (__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 90000)
// Thread local storage will be ignored because the linker for iOS 8 does not
// support it.
#define LITE_THREAD_LOCAL
#elif defined(LITE_WITH_SW)
// sw does not support thread_local
#define LITE_THREAD_LOCAL
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#define LITE_THREAD_LOCAL thread_local
#elif defined(_WIN32)
// The MSVC compiler does not support standards switches for C++11.
#define LITE_THREAD_LOCAL thread_local
#else
#error "[Paddle-Lite] C++11 support is required for paddle-lite compilation."
#endif
