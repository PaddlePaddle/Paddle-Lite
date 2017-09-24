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

#ifdef ANDROID
#include <android/log.h>
#endif

#ifndef MDL_MAC
#include <arm_neon.h>
#endif

#include "json/json11.h"
#include "math/math.h"
#include "commons/exception.h"

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
                            jclass exception_class = env->FindClass("com/baidu/graph/sdk/autoscanner/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         } catch (const std::exception &exception) {                                                 \
                            const char *message = (mdl::exception_prefix + exception.what()).c_str();                \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/graph/sdk/autoscanner/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         } catch (...) {                                                                             \
                            const char *message = (mdl::exception_prefix + "Unknown Exception.").c_str();            \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/graph/sdk/autoscanner/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         }
#endif

#endif
