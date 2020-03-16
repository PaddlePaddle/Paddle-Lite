/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stdio.h>
// #include <condition_variable>
// #include <mutex>

namespace paddle {
namespace zynqmp {

class FpgaIO {
 public:
  static FpgaIO& get_instance() {
    static FpgaIO s_instance;
    return s_instance;
  }

  void allocData(size_t s) { data_ = new float[s]; }

  float* getData() { return data_; }

  // void setMutex(std::mutex* mtx);
  // void setConditionVariable(std::condition_variable* condition);
  // void lock();
  // void unlock();

 private:
  std::mutex* mtx_ = nullptr;
  std::condition_variable* condition_ = nullptr;
  bool locked_ = false;

  float* data_ = nullptr;

  FpgaIO();
};
}  // namespace zynqmp
}  // namespace paddle
