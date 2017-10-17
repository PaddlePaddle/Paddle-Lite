#ifndef MDL_COMMONS_H
#define MDL_COMMONS_H
/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
/** The following copyright is cited from Caffe as an acknowledgement for its inspiring framework.
COPYRIGHT

All contributions by the University of California:
Copyright (c) 2014-2017 The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014-2017, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright over
their contributions to Caffe. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTRIBUTION AGREEMENT

By contributing to the BVLC/caffe repository through pull-request, comment,
or otherwise, the contributor releases their content to the
license and copyright terms herein.
================================================================================*/

#include <map>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

// #define NEED_DUMP true
#define MULTI_THREAD true
/**
 * This is an empirical value indicating how many inception layers could be accelerated by multi-thread.
 */
#define MAX_INCEPTION_NUM  9

#ifdef ANDROID
#include <android/log.h>
#include "math/neon_mathfun.h"
#endif

#ifndef MDL_MAC
#include <arm_neon.h>
#endif

#include "json/json11.h"
#include "math/math.h"
#include "commons/exception.h"

#ifdef MDL_LINUX
#include <string.h>
#include <limits.h>
#endif

using std::min;
using std::max;
using std::map;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::stringstream;

using Json = json11::Json;

using Math = mdl::Math;

using Time = decltype(std::chrono::high_resolution_clock::now());

using MDLException = mdl::MDLException;


namespace mdl {
    extern const char *log_tag;

    extern const int string_size;

    extern const int model_version;

    extern const string matrix_name_data;

    extern const string matrix_name_test_data;

    void im2col(const float *data_im, const int channels, const int height, const int width, const int kernel_size,
                const int pad, const int stride, float *data_col);

    Time time();

    double time_diff(Time t1, Time t2);

    void idle(const char *fmt, ...);

    bool equal(float a, float b);

    void copy(int length, float* x, float* y);



};

#ifdef ANDROID
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARNING, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#else
#define LOGI(...) mdl::idle(__VA_ARGS__);
#define LOGW(...) mdl::idle(__VA_ARGS__);
#define LOGD(...) mdl::idle(__VA_ARGS__);
#define LOGE(...) mdl::idle(__VA_ARGS__);
#endif

/**
 * throw the c++ exception to java level
 */
#ifdef ANDROID
#define EXCEPTION_HEADER try {
#define EXCEPTION_FOOTER } catch (const MDLException &exception) {                                                   \
                            const char *message = exception.what();                                                  \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/mdl/demo/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         } catch (const std::exception &exception) {                                                 \
                            const char *message = (mdl::exception_prefix + exception.what()).c_str();                \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/mdl/demo/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         } catch (...) {                                                                             \
                            const char *message = (mdl::exception_prefix + "Unknown Exception.").c_str();            \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/mdl/demo/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         }
#endif

#endif
