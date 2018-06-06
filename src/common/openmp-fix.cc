/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_MOBILE_USE_OPENMP
/**
 * android-ndk-r17 has a problem when linking with openmp.
 * if paddle-mobile enables -fopenmp, but didn't use those omp_* functions, after
 * linking another binary with libpaddle-mobile.so, the omp_get_thread_num will not work.
 * see test/common/test_openmp.cc
 * the detailed reason is still unclear, but this trick will work.
 * a better solution is hacking the linker, try some flags to make it link omp_* functions,
 * but I didn't find out how to make it work.
 */
#include <omp.h>
static int _ = omp_get_num_procs();
#endif
