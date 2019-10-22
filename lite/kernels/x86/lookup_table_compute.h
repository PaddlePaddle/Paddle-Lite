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

#include <iostream>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/fluid/eigen.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

/*struct LookupTableTimer {
  std::chrono::time_point<std::chrono::high_resolution_clock> timer_{};
  uint64_t total_{};

  void Start() { timer_ = std::chrono::high_resolution_clock::now(); }
  void Stop() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - timer_);
    Log(duration.count());
  }
  void Log(uint32_t timespan) { total_ += timespan; }
  ~LookupTableTimer() {
    LOG(INFO) << "lookup table timer: [" << total_ << "us]";
  }
};*/

template <typename T>
class LookupTableCompute : public KernelLite<TARGET(kX86), PRECISION(kInt64)> {
 public:
  using param_t = operators::LookupTableParam;

  void Run() override {
    auto &param = *param_.get_mutable<operators::LookupTableParam>();
    // auto& context = context_->As<X86Context>();
    auto *ids_t = param.Ids;
    auto *output_t = param.Out;
    LOG(INFO) << "lookup_table input ids tensor address: " << ids_t;
    int64_t padding_idx = param.padding_idx;
    auto *ids = ids_t->data<int64_t>();
    LOG(INFO) << "ids data address: " << ids;
    int64_t ids_numel = ids_t->dims().production();
    std::cout << "ids_numel: " << ids_numel << std::endl;
    std::cout << "ids tensor info: [";
    for (size_t i = 0; i < ids_numel; ++i) {
      std::cout << ids[i] << ",";
    }
    std::cout << "]" << std::endl;

    auto *table_t = param.W;
    int64_t row_number = table_t->dims()[0];
    int64_t row_width = table_t->dims()[1];
    std::cout << "row_number: " << row_number << std::endl;
    std::cout << "row_width: " << row_width << std::endl;

    auto *table = table_t->data<float>();
    auto *output = output_t->mutable_data<float>();
    memset(output, 0, output_t->dims().production() * sizeof(T));
    for (int64_t i = 0; i < ids_numel; ++i) {
      if (padding_idx != -1 && ids[i] == padding_idx) {
        memset(output + i * row_width, 0, row_width * sizeof(float));
      } else {
        std::cout << "*************************" << std::endl;
        std::cout << "ids[i]: " << ids[i] << std::endl;
        std::cout << "row_number: " << row_number << std::endl;
        std::cout << "*************************" << std::endl;
        CHECK_LT(ids[i], row_number);
        CHECK_GE(ids[i], 0);
        memcpy(output + i * row_width,
               table + ids[i] * row_width,
               row_width * sizeof(float));
      }
    }
  }

  virtual ~LookupTableCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
