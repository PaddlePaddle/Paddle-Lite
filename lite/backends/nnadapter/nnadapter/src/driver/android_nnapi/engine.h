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
#include "driver/android_nnapi/utility.h"

namespace nnadapter {
namespace android_nnapi {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();
  bool relax_fp32_to_fp16() { return relax_fp32_to_fp16_; }
  bool only_use_acc_device() { return only_use_acc_device_; }
  bool disable_cpu_device() { return disable_cpu_device_; }
  std::vector<ANeuralNetworksDevice*>* selected_devices() {
    return &selected_devices_;
  }

 private:
  void* device_{nullptr};
  bool relax_fp32_to_fp16_{true};
  bool only_use_acc_device_{false};
  bool disable_cpu_device_{false};
  std::vector<ANeuralNetworksDevice*> selected_devices_;
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Validate(const core::Model* model, bool* supported_operations);
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
  // Map NNAdapter operand to NNAPI operand index
  std::map<core::Operand*, std::vector<uint32_t>> operand_indexes_;
  std::vector<void*> operand_buffers_;
  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
  ANeuralNetworksExecution* execution_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
};

}  // namespace android_nnapi
}  // namespace nnadapter
