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
#include "../../nnadapter_driver.h"   // NOLINT
#include "HiAiModelManagerService.h"  // NOLINT
#include "hiai_ir_build.h"            // NOLINT

namespace nnadapter {
namespace driver {
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

  int Build(driver::Model* model, driver::Cache* cache);
  int Execute(uint32_t input_count,
              driver::Argument* input_arguments,
              uint32_t output_count,
              driver::Argument* output_arguments);

 private:
  // Operation converters
  std::shared_ptr<ge::Operator> ConvertOperand(driver::Operand* operand);
  int ConvertConv2D(driver::Operation* operation);
  int ConvertFullyConnected(driver::Operation* operation);
  int ConvertAverageAndMaxPool2D(driver::Operation* operation);
  int ConvertElementwiseBinaryOperations(driver::Operation* operation);
  int ConvertSoftmax(driver::Operation* operation);
  int ConvertActivationUnaryOperations(driver::Operation* operation);

 private:
  Context* context_{nullptr};
  // NNAdapter operand to hiai nodes
  std::map<driver::Operand*, std::vector<std::shared_ptr<ge::Operator>>> nodes_;
  std::string model_name_{""};
  std::shared_ptr<hiai::AiModelMngerClient> model_client_{nullptr};
  std::vector<std::shared_ptr<hiai::AiTensor>> input_tensors_{};
  std::vector<std::shared_ptr<hiai::AiTensor>> output_tensors_{};
};

}  // namespace huawei_kirin_npu
}  // namespace driver
}  // namespace nnadapter
