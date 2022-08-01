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

#include <limits.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "driver/huawei_ascend_npu/utility.h"
#include "op_proto/built-in/inc/all_ops.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_ascend_npu {

class Device {
 public:
  Device();
  ~Device();
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  int first_device_id() {
    return selected_device_ids_.empty() ? 0 : selected_device_ids_[0];
  }
  AscendConfigParams* ascend_config_params() { return &ascend_config_params_; }
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  std::vector<int> selected_device_ids_;
  AscendConfigParams ascend_config_params_;
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
  std::shared_ptr<AclModelClient> model_client_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  DynamicShapeMode dynamic_shape_mode_{DYNAMIC_SHAPE_MODE_NONE};
};

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
