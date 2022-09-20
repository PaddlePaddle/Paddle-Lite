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
#include "driver/verisilicon_timvx/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();
  double batchnorm_fusion_max_allowed_quant_scale_deviation() {
    return batchnorm_fusion_max_allowed_quant_scale_deviation_;
  }

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  double batchnorm_fusion_max_allowed_quant_scale_deviation_{0.0f};
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
  // Map NNAdapter operand to tim-vx tensor
  std::map<core::Operand*, std::vector<std::shared_ptr<tim::vx::Tensor>>>
      tensors_;
  std::shared_ptr<tim::vx::Graph> graph_{nullptr};
  std::shared_ptr<tim::vx::Context> ctx_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> output_tensors_;
};

}  // namespace verisilicon_timvx
}  // namespace nnadapter
