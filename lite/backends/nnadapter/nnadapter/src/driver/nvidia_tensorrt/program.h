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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "driver/nvidia_tensorrt/calibrator.h"
#include "driver/nvidia_tensorrt/kernel/cuda/special_softmax.h"
#include "driver/nvidia_tensorrt/kernel/host/naive_softmax.h"
#include "driver/nvidia_tensorrt/kernel/kernel.h"
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
  explicit Context(void* device,
                   const char* properties,
                   int (*callback)(int event_id, void* user_data));
  ~Context() {}

  nvinfer1::DeviceType DeviceType() { return device_type_; }
  int DeviceId() { return device_id_; }
  PrecisionMode Precision() { return precision_; }
  bool GpuFallback() { return gpu_fallback_; }
  std::string CalibrationDatasetPath() { return calibration_dataset_path_; }
  std::string CalibrationTablePath() { return calibration_table_path_; }
  size_t MaxWorkspaceSize() { return max_workspce_size_; }
  cudaStream_t CudaStream();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  int (*callback_)(int event_id, void* user_data){nullptr};  // NOLINT
  nvinfer1::DeviceType device_type_{nvinfer1::DeviceType::kGPU};
  int device_id_{0};
  PrecisionMode precision_{kFloat32};
  bool gpu_fallback_{true};
  std::string calibration_dataset_path_;
  std::string calibration_table_path_;
  size_t max_workspce_size_{0};
};

class ProgramBase {
 public:
  ProgramBase() {}
  virtual ~ProgramBase() {}

  virtual int Build() = 0;
  virtual int Execute(std::vector<std::shared_ptr<Tensor>>* input_tensors,
                      std::vector<std::shared_ptr<Tensor>>* output_tensors,
                      cudaStream_t stream) = 0;
};

class TensorrtProgram : public ProgramBase {
 public:
  explicit TensorrtProgram(Context* context,
                           core::Model* model,
                           std::vector<uint8_t>* cache)
      : context_(context), model_(model), cache_(cache) {}
  ~TensorrtProgram() { Clear(); }

  int Build();
  int Execute(std::vector<std::shared_ptr<Tensor>>* input_tensors,
              std::vector<std::shared_ptr<Tensor>>* output_tensors,
              cudaStream_t stream);

 private:
  void Clear();
  void CompleteConfig();
  // Build model and save to plan_
  int BuildFromModel();
  // Read model from cache to plan_
  int BuildFromCache();

 private:
  Context* context_{nullptr};
  core::Model* model_{nullptr};
  std::vector<uint8_t>* cache_{nullptr};
  std::unique_ptr<nvinfer1::IBuilder, TensorrtDeleter> builder_;
  std::unique_ptr<nvinfer1::INetworkDefinition, TensorrtDeleter> network_;
  std::unique_ptr<nvinfer1::IBuilderConfig, TensorrtDeleter> config_;
  std::unique_ptr<nvinfer1::IHostMemory, TensorrtDeleter> plan_;
  std::unique_ptr<nvinfer1::IRuntime, TensorrtDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TensorrtDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TensorrtDeleter>
      execution_context_;
  std::unique_ptr<Int8EntropyCalibrator> calibrator_;
  std::map<core::Operand*, std::vector<nvinfer1::ITensor*>> tensors_;
  std::vector<int> input_indices_;
  std::vector<int> output_indices_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  int max_batch_size_{1};
  bool with_dynamic_shape_{false};
};

class CudaProgram : public ProgramBase {
 public:
  explicit CudaProgram(Context* context,
                       core::Model* model,
                       std::vector<uint8_t>* cache)
      : context_(context), model_(model), cache_(cache) {}
  ~CudaProgram() { Clear(); }

  int Build();
  int Execute(std::vector<std::shared_ptr<Tensor>>* input_tensors,
              std::vector<std::shared_ptr<Tensor>>* output_tensors,
              cudaStream_t stream);

 private:
  void Clear();
  int BuildFromModel();
  int BuildFromCache();

 private:
  Context* context_{nullptr};
  core::Model* model_{nullptr};
  std::vector<uint8_t>* cache_{nullptr};
  std::vector<core::Operation*> operations_;
  std::vector<std::shared_ptr<KernelBase>> kernels_;
  std::map<core::Operand*, std::shared_ptr<Tensor>> tensors_;
};

class HostProgram : public ProgramBase {
 public:
  explicit HostProgram(Context* context,
                       core::Model* model,
                       std::vector<uint8_t>* cache)
      : context_(context), model_(model), cache_(cache) {}
  ~HostProgram() { Clear(); }

  int Build();
  int Execute(std::vector<std::shared_ptr<Tensor>>* input_tensors,
              std::vector<std::shared_ptr<Tensor>>* output_tensors,
              cudaStream_t stream);

 private:
  void Clear();
  int BuildFromModel();
  int BuildFromCache();

 private:
  Context* context_{nullptr};
  core::Model* model_{nullptr};
  std::vector<uint8_t>* cache_{nullptr};
  std::vector<core::Operation*> operations_;
  std::vector<std::shared_ptr<KernelBase>> kernels_;
  std::map<core::Operand*, std::shared_ptr<Tensor>> tensors_;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
