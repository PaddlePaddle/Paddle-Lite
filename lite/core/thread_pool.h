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
#include <atomic>
#include <condition_variable>  //NOLINT
#include <functional>
#include <mutex>   //NOLINT
#include <thread>  //NOLINT
#include <tuple>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {

class ThreadPool {
 public:
  typedef std::function<void(int, int)> TASK;
  typedef std::pair<std::function<void(int, int)>, int> TASK_BASIC;
  typedef std::tuple<std::function<void(int, int)>, int, int, int> TASK_COMMON;

  static void Enqueue(TASK_BASIC&& task);
  static void Enqueue(TASK_COMMON&& task);
  static void AcquireThreadPool();
  static void ReleaseThreadPool();
  static int Init(int number);
  static void Destroy();

 private:
  static ThreadPool* gInstance;
  explicit ThreadPool(int number = 0);
  ~ThreadPool();

  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  bool ready_{true};
  std::pair<TASK, std::vector<std::atomic<bool>*>> tasks_;
  std::condition_variable cv_;
  std::mutex mutex_;

  int thread_num_ = 0;
};
}  // namespace lite
}  // namespace paddle
