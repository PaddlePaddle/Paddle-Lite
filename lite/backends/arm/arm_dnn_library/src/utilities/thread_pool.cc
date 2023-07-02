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

#include "utilities/thread_pool.h"
#include <string.h>
#include "utilities/logging.h"

namespace armdnnlibrary {

ThreadPool* ThreadPool::g_ThreadPoolInstance = nullptr;
static std::mutex
    g_ThreadPoolInitMutex;  // confirm thread-safe when use singleton mode
int ThreadPool::Initialize(int thread_num) {
  // Don't instantiate ThreadPool when compile ThreadPool and only use 1 thread.
  if (thread_num <= 1) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(g_ThreadPoolInitMutex);
  if (nullptr == g_ThreadPoolInstance) {
    g_ThreadPoolInstance = new ThreadPool(thread_num);
  }
  return g_ThreadPoolInstance->thread_num_;
}

void ThreadPool::Destroy() {
  std::lock_guard<std::mutex> lock(g_ThreadPoolInitMutex);
  if (nullptr != g_ThreadPoolInstance) {
    delete g_ThreadPoolInstance;
    g_ThreadPoolInstance = nullptr;
  }
}

ThreadPool::ThreadPool(int thread_num) {
  thread_num_ = thread_num;
  for (int i = 0; i < thread_num_; ++i) {
    tasks_.second.emplace_back(new std::atomic<bool>{false});
  }
  for (int thread_index = 1; thread_index < thread_num_; ++thread_index) {
    workers_.emplace_back([this, thread_index]() {
      while (!stop_) {
        while (!(*tasks_.second[thread_index])) {
          std::this_thread::yield();
        }
        tasks_.first(thread_index, thread_index);
        *tasks_.second[thread_index] = false;
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  stop_ = true;
  for (auto& worker : workers_) {
    worker.join();
  }
  for (auto c : tasks_.second) {
    delete c;
  }
}

void ThreadPool::Acquire() {
  if (nullptr == g_ThreadPoolInstance) {
    return;
  }
  ARM_DNN_LIBRARY_VLOG(5) << "ThreadPool::Acquire()\n";
  std::unique_lock<std::mutex> lock(g_ThreadPoolInstance->mutex_);
  while (!g_ThreadPoolInstance->ready_) g_ThreadPoolInstance->cv_.wait(lock);
  g_ThreadPoolInstance->ready_ = false;
  return;
}

void ThreadPool::Release() {
  if (nullptr == g_ThreadPoolInstance) {
    return;
  }
  ARM_DNN_LIBRARY_VLOG(5) << "ThreadPool::Release()\n";
  std::unique_lock<std::mutex> lock(g_ThreadPoolInstance->mutex_);
  g_ThreadPoolInstance->ready_ = true;
  g_ThreadPoolInstance->cv_.notify_all();
}

void ThreadPool::Enqueue(SIMPLE_TASK&& task) {
  if (task.second <= 1 || (nullptr == g_ThreadPoolInstance)) {
    for (int i = 0; i < task.second; ++i) {
      task.first(i, 0);
    }
    return;
  }
  int work_size = task.second;
  if (work_size > g_ThreadPoolInstance->thread_num_) {
    g_ThreadPoolInstance->tasks_.first = [work_size, &task](int index,
                                                            int tId) {
      for (int v = tId; v < work_size; v += g_ThreadPoolInstance->thread_num_) {
        task.first(v, tId);  // nested lambda func
      }
    };
    work_size = g_ThreadPoolInstance->thread_num_;
  } else {
    g_ThreadPoolInstance->tasks_.first = std::move(task.first);
  }
  for (int i = 1; i < work_size; ++i) {
    *(g_ThreadPoolInstance->tasks_.second[i]) = true;
  }
  // Invoke tid 0 callback in main thread, other tid task is invoked in child
  // thread
  g_ThreadPoolInstance->tasks_.first(0, 0);
  bool complete = true;
  // Check tid 1 to thread_num - 1 all work completed in child thread
  do {
    std::this_thread::yield();
    complete = true;
    for (int i = 1; i < work_size; ++i) {
      if (*g_ThreadPoolInstance->tasks_.second[i]) {
        complete = false;
        break;
      }
    }
  } while (!complete);
}

void ThreadPool::Enqueue(COMMON_TASK&& task) {
  int end = std::get<1>(task);
  int start = std::get<2>(task);
  int step = std::get<3>(task);
  int work_size = (end - start + step - 1) / step;
  if (work_size <= 1 || (nullptr == g_ThreadPoolInstance)) {
    for (int v = start; v < end; v += step) {
      std::get<0>(task)(v, 0);
    }
    return;
  }
  if (work_size > g_ThreadPoolInstance->thread_num_) {
    g_ThreadPoolInstance->tasks_.first = ([=, &task](int index, int tid) {
      auto start_index = start + tid * step;
      auto stride = g_ThreadPoolInstance->thread_num_ * step;
      for (int v = start_index; v < end; v += stride) {
        std::get<0>(task)(v, tid);  // nested lambda func
      }
    });
    work_size = g_ThreadPoolInstance->thread_num_;
  } else {
    g_ThreadPoolInstance->tasks_.first = ([=, &task](int index, int tid) {
      auto v = start + tid * step;
      std::get<0>(task)(v, tid);  // nested lambda func
    });
  }
  for (int i = 1; i < work_size; ++i) {
    *(g_ThreadPoolInstance->tasks_.second[i]) = true;
  }
  // Invoke tid 0 callback in main thread, other tid task is invoked in new
  // thread
  g_ThreadPoolInstance->tasks_.first(0, 0);
  bool complete = true;
  // Check tid 1 to thread_num - 1 all work completed in new thread
  do {
    std::this_thread::yield();
    complete = true;
    for (int i = 1; i < work_size; ++i) {
      if (*g_ThreadPoolInstance->tasks_.second[i]) {
        complete = false;
        break;
      }
    }
  } while (!complete);
}

}  // namespace armdnnlibrary
