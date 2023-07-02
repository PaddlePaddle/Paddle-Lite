// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <condition_variable>  // NOLINT
#include <functional>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

namespace armdnnlibrary {

class ThreadPool {
 public:
  typedef std::function<void(int, int)> TASK;
  typedef std::pair<std::function<void(int, int)>, int> SIMPLE_TASK;
  typedef std::tuple<std::function<void(int, int)>, int, int, int> COMMON_TASK;

  static void Enqueue(SIMPLE_TASK&& task);
  static void Enqueue(COMMON_TASK&& task);
  static void Acquire();
  static void Release();
  static int Initialize(int thread_num);
  static void Destroy();

 private:
  static ThreadPool* g_ThreadPoolInstance;
  explicit ThreadPool(int thread_num = 0);
  ~ThreadPool();

  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  bool ready_{true};
  std::pair<TASK, std::vector<std::atomic<bool>*>> tasks_;
  std::condition_variable cv_;
  std::mutex mutex_;
  int thread_num_ = 0;
};

}  // namespace armdnnlibrary

#ifdef ARM_DNN_LIBRARY_WITH_THREAD_POOL
/* Support simple task
 * for (int i = 0; i < work_size; ++i)
 */
#define ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(index, tid, work_size) \
  {                                                                          \
    std::pair<std::function<void(int, int)>, int> task;                      \
    task.second = work_size;                                                 \
  task.first = [&](int index, int tid) {
#define ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_END()  \
  }                                                    \
  ;                                                    \
  armdnnlibrary::ThreadPool::Enqueue(std::move(task)); \
  }

/* Support common task
 * for (int i = start; i < end; i += step)
 */
#define ARM_DNN_LIBRARY_THREAD_POOL_COMMON_TASK_BEGIN(             \
    index, tid, end, start, step)                                  \
  {                                                                \
    std::tuple<std::function<void(int, int)>, int, int, int> task; \
    std::get<3>(task) = step;                                      \
    std::get<2>(task) = start;                                     \
    std::get<1>(task) = end;                                       \
  std::get<0>(task) = [&](int index, int tid) {
#define ARM_DNN_LIBRARY_THREAD_POOL_COMMON_TASK_END()  \
  }                                                    \
  ;                                                    \
  armdnnlibrary::ThreadPool::Enqueue(std::move(task)); \
  }
#elif defined(ARM_DNN_LIBRARY_WITH_OMP)
#include <omp.h>
#define ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(index, tid, work_size) \
  _Pragma("omp parallel for") for (int index = 0; index < (work_size);       \
                                   ++index) {
#define ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_END() }

#define ARM_DNN_LIBRARY_THREAD_POOL_COMMON_TASK_BEGIN(                 \
    index, tid, end, start, step)                                      \
  _Pragma("omp parallel for") for (int index = (start); index < (end); \
                                   index += (step)) {
#define ARM_DNN_LIBRARY_THREAD_POOL_COMMON_TASK_END() }
#else
#define ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_BEGIN(index, tid, work_size) \
  for (int index = 0; index < (work_size); ++index) {
#define ARM_DNN_LIBRARY_THREAD_POOL_SIMPLE_TASK_END() }

#define ARM_DNN_LIBRARY_THREAD_POOL_COMMON_TASK_BEGIN( \
    index, tid, end, start, step)                      \
  for (int index = (start); index < (end); index += (step)) {
#define ARM_DNN_LIBRARY_THREAD_POOL_COMMON_TASK_END() }
#endif
