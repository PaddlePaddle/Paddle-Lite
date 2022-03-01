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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "driver/huawei_kirin_npu/utility.h"
#include "graph/op/all_ops.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_kirin_npu {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();

  int freq_level() { return freq_level_; }
  int framework_type() { return framework_type_; }
  int model_type() { return model_type_; }
  int device_type() { return device_type_; }

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  int freq_level_{3};
  int framework_type_{0};
  int model_type_{0};
  int device_type_{0};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(core::Model* model, core::Cache* cache);
  int Execute(uint32_t input_count,
              core::Argument* input_arguments,
              uint32_t output_count,
              core::Argument* output_arguments);

 private:
  void Clear();
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to GE operator
  std::map<core::Operand*, std::vector<std::shared_ptr<Operator>>> operators_;
  std::string model_name_{""};
  std::shared_ptr<hiai::AiModelMngerClient> model_client_{nullptr};
  std::vector<std::shared_ptr<hiai::AiTensor>> input_tensors_{};
  std::vector<std::shared_ptr<hiai::AiTensor>> output_tensors_{};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
};

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
