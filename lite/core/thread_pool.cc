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

#include "lite/core/thread_pool.h"
#include <string.h>
#include "lite/utils/log/logging.h"

namespace paddle {
namespace lite {

ThreadPool* ThreadPool::gInstance = nullptr;
static std::mutex gInitMutex;  // confirm thread-safe when use singleton mode
int ThreadPool::Init(int number) {
  // Don't instantiate ThreadPool when compile ThreadPool and only use 1 thread
  if (number <= 1) {
    return 1;
  }
  std::lock_guard<std::mutex> _l(gInitMutex);
  if (nullptr == gInstance) {
    gInstance = new ThreadPool(number);
  }
  return gInstance->thread_num_;
}
void ThreadPool::Destroy() {
  std::lock_guard<std::mutex> _l(gInitMutex);
  if (nullptr != gInstance) {
    delete gInstance;
    gInstance = nullptr;
  }
}

ThreadPool::ThreadPool(int number) {
  thread_num_ = number;
  for (int i = 0; i < thread_num_; ++i) {
    tasks_.second.emplace_back(new std::atomic<bool>{false});
  }
  for (int thread_index = 1; thread_index < thread_num_; ++thread_index) {
    workers_.emplace_back([this, thread_index]() {
      while (!stop_) {
        // if (*tasks_.second[thread_index]) {
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

void ThreadPool::AcquireThreadPool() {
  if (nullptr == gInstance) {
    return;
  }
  LOG(INFO) << "ThreadPool::AcquireThreadPool()\n";
  std::unique_lock<std::mutex> _l(gInstance->mutex_);
  while (!gInstance->ready_) gInstance->cv_.wait(_l);
  gInstance->ready_ = false;
  return;
}

void ThreadPool::ReleaseThreadPool() {
  if (nullptr == gInstance) {
    return;
  }
  LOG(INFO) << "ThreadPool::ReleaseThreadPool()\n";
  std::unique_lock<std::mutex> _l(gInstance->mutex_);
  gInstance->ready_ = true;
  gInstance->cv_.notify_all();
}

void ThreadPool::Enqueue(TASK_BASIC&& task) {
  if (task.second <= 1 || (nullptr == gInstance)) {
    for (int i = 0; i < task.second; ++i) {
      task.first(i, 0);
    }
    return;
  }
  int work_size = task.second;
  if (work_size > gInstance->thread_num_) {
    gInstance->tasks_.first = [work_size, &task](int index, int tId) {
      for (int v = tId; v < work_size; v += gInstance->thread_num_) {
        task.first(v, tId);  // nested lambda func
      }
    };
    work_size = gInstance->thread_num_;
  } else {
    gInstance->tasks_.first = std::move(task.first);
  }
  for (int i = 1; i < work_size; ++i) {
    *(gInstance->tasks_.second[i]) = true;
  }
  // invoke tid 0 callback in main thread
  // other tid task is invoked in child thread
  gInstance->tasks_.first(0, 0);
  bool complete = true;
  // check tid 1 to thread_num - 1 all work completed in child thread
  do {
    std::this_thread::yield();
    complete = true;
    for (int i = 1; i < work_size; ++i) {
      if (*gInstance->tasks_.second[i]) {
        complete = false;
        break;
      }
    }
  } while (!complete);
}

void ThreadPool::Enqueue(TASK_COMMON&& task) {
  int end = std::get<1>(task);
  int start = std::get<2>(task);
  int step = std::get<3>(task);
  int work_size = (end - start + step - 1) / step;
  if (work_size <= 1 || (nullptr == gInstance)) {
    for (int v = start; v < end; v += step) {
      std::get<0>(task)(v, 0);
    }
    return;
  }
  if (work_size > gInstance->thread_num_) {
    gInstance->tasks_.first = ([=, &task](int index, int tId) {
      auto start_index = start + tId * step;
      auto stride = gInstance->thread_num_ * step;
      for (int v = start_index; v < end; v += stride) {
        std::get<0>(task)(v, tId);  // nested lambda func
      }
    });
    work_size = gInstance->thread_num_;
  } else {
    gInstance->tasks_.first = ([=, &task](int index, int tId) {
      auto v = start + tId * step;
      std::get<0>(task)(v, tId);  // nested lambda func
    });
  }
  for (int i = 1; i < work_size; ++i) {
    *(gInstance->tasks_.second[i]) = true;
  }
  // invoke tid 0 callback in main thread
  // other tid task is invoked in new thread
  gInstance->tasks_.first(0, 0);
  bool complete = true;
  // check tid 1 to thread_num - 1 all work completed in new thread
  do {
    std::this_thread::yield();
    complete = true;
    for (int i = 1; i < work_size; ++i) {
      if (*gInstance->tasks_.second[i]) {
        complete = false;
        break;
      }
    }
  } while (!complete);
}

}  // namespace lite
}  // namespace paddle
