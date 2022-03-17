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

  nvinfer1::DeviceType DeviceType() { return device_type_; }
  int DeviceId() { return device_id_; }
  PrecisionMode Precision() { return precision_; }
  bool GpuFallback() { return gpu_fallback_; }

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  nvinfer1::DeviceType device_type_{nvinfer1::DeviceType::kGPU};
  int device_id_{0};
  PrecisionMode precision_{kFloat32};
  bool gpu_fallback_{true};
};

struct Deleter {
  template <typename T>
  void operator()(T* obj) const {
#if TENSORRT_MAJOR_VERSION >= 8
    delete obj;
#else
    if (obj) {
      obj->destroy();
    }
#endif
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
  void CompleteConfig(core::Model* model);
  // Build model and save to plan_
  int BuildFromModel(core::Model* model);
  // Read model from cache to plan_
  int BuildFromCache(core::Cache* cache);
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);

 private:
  std::unique_ptr<nvinfer1::IBuilder, Deleter> builder_;
  std::unique_ptr<nvinfer1::INetworkDefinition, Deleter> network_;
  std::unique_ptr<nvinfer1::IBuilderConfig, Deleter> config_;
  std::unique_ptr<nvinfer1::IHostMemory, Deleter> plan_;
  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> execution_context_;
  std::vector<std::shared_ptr<void>> device_buffers_;
  std::map<core::Operand*, std::vector<nvinfer1::ITensor*>> tensors_;
  std::vector<int> input_indices_;
  std::vector<int> output_indices_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  Context* context_{nullptr};
  bool with_dynamic_shape_{false};
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
