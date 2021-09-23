// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <tuple>
#include <utility>
#include "lite/core/thread_pool.h"

#ifdef LITE_USE_THREAD_POOL
/* support basic for loop
 * for (int i = 0; i < work_size; ++i)
 */
#define LITE_PARALLEL_BEGIN(index, tid, work_size)      \
  {                                                     \
    std::pair<std::function<void(int, int)>, int> task; \
    task.second = work_size;                            \
  task.first = [&](int index, int tid) {
#define LITE_PARALLEL_END()                           \
  }                                                   \
  ;                                                   \
  paddle::lite::ThreadPool::Enqueue(std::move(task)); \
  }

/* support common for loop
 * for (int i = start; i < end; i += step)
 */
#define LITE_PARALLEL_COMMON_BEGIN(index, tid, end, start, step)   \
  {                                                                \
    std::tuple<std::function<void(int, int)>, int, int, int> task; \
    std::get<3>(task) = step;                                      \
    std::get<2>(task) = start;                                     \
    std::get<1>(task) = end;                                       \
  std::get<0>(task) = [&](int index, int tid) {
#define LITE_PARALLEL_COMMON_END()                    \
  }                                                   \
  ;                                                   \
  paddle::lite::ThreadPool::Enqueue(std::move(task)); \
  }

#elif defined(ARM_WITH_OMP)
#include <omp.h>

#define LITE_PARALLEL_BEGIN(index, tid, work_size)                     \
  _Pragma("omp parallel for") for (int index = 0; index < (work_size); \
                                   ++index) {
#define LITE_PARALLEL_END() }

#define LITE_PARALLEL_COMMON_BEGIN(index, tid, end, start, step)       \
  _Pragma("omp parallel for") for (int index = (start); index < (end); \
                                   index += (step)) {
#define LITE_PARALLEL_COMMON_END() }

#else
#define LITE_PARALLEL_BEGIN(index, tid, work_size) \
  for (int index = 0; index < (work_size); ++index) {
#define LITE_PARALLEL_END() }

#define LITE_PARALLEL_COMMON_BEGIN(index, tid, end, start, step) \
  for (int index = (start); index < (end); index += (step)) {
#define LITE_PARALLEL_COMMON_END() }
#endif
