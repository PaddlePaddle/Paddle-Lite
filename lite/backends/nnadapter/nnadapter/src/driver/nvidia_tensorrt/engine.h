// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>
#include "driver/nvidia_tensorrt/utility.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
};

struct Deleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
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
  // Build from model or cache
  int BuildFromModel(core::Model* model);
  int BuildFromCache(core::Cache* cache);

 private:
  std::unique_ptr<nvinfer1::IBuilder, Deleter> builder_;
  std::unique_ptr<nvinfer1::INetworkDefinition, Deleter> network_;
  std::unique_ptr<nvinfer1::IBuilderConfig, Deleter> config_;
  std::unique_ptr<nvinfer1::IHostMemory, Deleter> plan_;
  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> nv_context_;
  std::vector<std::shared_ptr<void>> device_data_;
  std::map<core::Operand*, std::vector<nvinfer1::ITensor*>> tensors_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  Context* context_;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
