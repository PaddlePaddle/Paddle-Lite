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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/intel_openvino/utility.h"

namespace nnadapter {
namespace intel_openvino {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();
  std::string GetFirtSelectedDeviceName() const {
    return selected_device_names_[0];
  }
  std::shared_ptr<std::map<std::string, ov::AnyMap>> GetDeviceConfig() {
    return std::make_shared<std::map<std::string, ov::AnyMap>>(
        device_config_map_);
  }

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  std::vector<std::string> selected_device_names_{"CPU"};
  // Device config map.
  std::map<std::string, ov::AnyMap> device_config_map_;
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program() {}

  int Build(core::Model* model, core::Cache* cache);
  int Execute(uint32_t input_count,
              core::Argument* input_arguments,
              uint32_t output_count,
              core::Argument* output_arguments);

 private:
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);

 private:
  Context* context_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::shared_ptr<ov::Core> runtime_core_{nullptr};
  std::map<core::Operand*, std::vector<std::shared_ptr<Tensor>>> tensor_map_;
  std::map<core::Operand*, std::shared_ptr<default_opset::Parameter>>
      parameter_node_map_;
  std::vector<std::shared_ptr<Operator>> result_nodes_;
  std::shared_ptr<ov::CompiledModel> compiled_model_{nullptr};
  bool with_dynamic_shape_{false};
};

}  // namespace intel_openvino
}  // namespace nnadapter
