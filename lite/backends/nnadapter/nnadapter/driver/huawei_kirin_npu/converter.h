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
#include "driver/huawei_kirin_npu/utility.h"
#include "graph/compatible/all_ops.h"
#include "graph/compatible/operator_reg.h"
#include "utility/logging.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_kirin_npu {

class Context {
 public:
  Context();
  ~Context();

  int freq_level() { return freq_level_; }
  int framework_type() { return framework_type_; }
  int model_type() { return model_type_; }
  int device_type() { return device_type_; }

 private:
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

  int Build(hal::Model* model, hal::Cache* cache);
  int Execute(uint32_t input_count,
              hal::Argument* input_arguments,
              uint32_t output_count,
              hal::Argument* output_arguments);

 private:
  // Operand converters
  template <typename T>
  std::shared_ptr<T> AddOperator(
      hal::Operand* operand = nullptr /* anonymous operand */) {
    auto it = operators_.find(operand);
    if (it != operators_.end()) {
      // Only temporary variable or model output node can be shared with the
      // same operand
      if (typeid(T) == typeid(ge::op::Const) ||
          typeid(T) == typeid(ge::op::Data)) {
        NNADAPTER_LOG(FATAL)
            << "Duplicate mapping a non-temporary variable NNAdapter operand@0x"
            << std::hex << operand << " to a HiAI IR.";
        return nullptr;
      }
    } else {
      auto result = operators_.insert(std::make_pair(
          operand, std::vector<std::shared_ptr<ge::Operator>>()));
      NNADAPTER_CHECK(result.second);
      it = result.first;
    }
    int index = it->second.size();
    std::string name = string_format("@0x%X_%d", operand, index);
    auto op = std::make_shared<T>(name);
    it->second.push_back(op);
    return op;
  }
  std::shared_ptr<ge::Operator> ConvertOperand(
      hal::Operand* operand, std::vector<int64_t> dimensions = {});

  // Operation converters
  int ConvertConv2D(hal::Operation* operation);
  int ConvertFullyConnected(hal::Operation* operation);
  int ConvertPool2D(hal::Operation* operation);
  int ConvertElementwise(hal::Operation* operation);
  int ConvertSoftmax(hal::Operation* operation);
  int ConvertActivation(hal::Operation* operation);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to hiai operators
  std::map<hal::Operand*, std::vector<std::shared_ptr<ge::Operator>>>
      operators_;
  std::string model_name_{""};
  std::shared_ptr<hiai::AiModelMngerClient> model_client_{nullptr};
  std::vector<std::shared_ptr<hiai::AiTensor>> input_tensors_{};
  std::vector<std::shared_ptr<hiai::AiTensor>> output_tensors_{};
};

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
