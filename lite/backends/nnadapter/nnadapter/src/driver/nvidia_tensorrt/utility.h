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

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <memory>
#include <vector>
#include "core/types.h"
#include "driver/nvidia_tensorrt/operation/type.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

// Following environment variables are used to enable tensorrt features.
// Use which type of device to run, for example: "GPU", "DLA"
#define NVIDIA_TENSORRT_DEVICE_TYPE "NVIDIA_TENSORRT_DEVICE_TYPE"

// Use which device core, for example: "0", "1"
#define NVIDIA_TENSORRT_DEVICE_ID "NVIDIA_TENSORRT_DEVICE_ID"

// Use which precision, for example: "float32", "float16", "int8"
#define NVIDIA_TENSORRT_PRECISION "NVIDIA_TENSORRT_PRECISION"

// Whether to allow gpu fallback, for example: "0", "1"
#define NVIDIA_TENSORRT_GPU_FALLBACK "NVIDIA_TENSORRT_GPU_FALLBACK"

// Path to calibration dataset, for example: "/home/demo/dataset"
// The path containers following files: "list.txt", "input0_0",...
#define NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH \
  "NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH"

// Path to calibration table, for example:
// "/home/demo/dataset/calibration_table"
#define NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH \
  "NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH"

// Max workspace size, for example: "1048576"
#define NVIDIA_TENSORRT_MAX_WORKSPACE_SIZE "NVIDIA_TENSORRT_MAX_WORKSPACE_SIZE"

#define NVIDIA_TENSORRT_GET_EXTERNAL_CUDA_STREAM 0x0100

#define TENSORRT_VERSION_GE(major, minor, patch, build)          \
  (((NV_TENSORRT_MAJOR > major) ||                               \
    (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR > minor) || \
    (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
     NV_TENSORRT_PATCH > patch) ||                               \
    (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
     NV_TENSORRT_PATCH == patch && NV_TENSORRT_BUILD >= build)))

#if TENSORRT_VERSION_GE(8, 0, 0, 0)
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

// Supported places
typedef enum {
  kTensorrt = 0,
  kCUDA = 1,
  kHost = 2,
} DeviceType;

// Tensorrt precision mode options
typedef enum {
  kFloat32 = 0,
  kFloat16 = 1,
  kInt8 = 2,
} PrecisionMode;

struct TensorrtDeleter {
  template <typename T>
  void operator()(T* obj) const {
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
    delete obj;
#else
    if (obj) {
      obj->destroy();
    }
#endif
  }
};

struct HostMemoryDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr) {
      free(ptr);
    }
  }
};

class Tensor {
 public:
  Tensor() {}
  ~Tensor();

  void SetData(void* cuda_buffer,
               const std::vector<int32_t>& dims,
               nvinfer1::DataType data_type);

  // Only support copy from cuda to host
  void* Data(bool return_cuda_buffer = true);

  void Resize(const std::vector<int32_t>& dims) { dims_ = dims; }

  std::vector<int32_t> Dims() { return dims_; }

  uint32_t Length();

  void SetDataType(nvinfer1::DataType data_type) { data_type_ = data_type; }

  nvinfer1::DataType DateType() { return data_type_; }

 private:
  void* cuda_buffer_{nullptr};
  uint32_t cuda_buffer_length_{0};
  bool own_cuda_buffer_{true};
  std::unique_ptr<void, HostMemoryDeleter> host_buffer_;
  uint32_t host_buffer_length_{0};
  nvinfer1::DataType data_type_{nvinfer1::DataType::kFLOAT};
  std::vector<int32_t> dims_;
};

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) TRT_NOEXCEPT override;

  static TrtLogger* Global() TRT_NOEXCEPT {
    static TrtLogger* logger = new TrtLogger;
    return logger;
  }
};

// Convert NNAdapter types to nvinfer1::DataType
nvinfer1::DataType ConvertToNVDataType(
    NNAdapterOperandPrecisionCode input_precision);

uint32_t GetNVTypeSize(nvinfer1::DataType type);

template <typename T>
nvinfer1::DataType GetNVDateType();

template <typename T>
size_t SerializedSize(const T value);

template <typename T>
size_t SerializedSize(const std::vector<T>& value);

template <typename T>
void Serialize(void** buffer, const T value);

template <typename T>
void Serialize(void** buffer, const std::vector<T>& value);

template <typename T>
void Deserialize(const void** buffer, size_t* buffer_size, T* value);

template <typename T>
void Deserialize(const void** buffer,
                 size_t* buffer_size,
                 std::vector<T>* value);

// Only remain min/opt/max shapes
void ConvertDynamicDimensions(NNAdapterOperandDimensionType* dimensions);

int GetMaxBatchSize(const NNAdapterOperandDimensionType& dimensions);

core::Argument* FindArgumentByIndex(core::Argument* arguments,
                                    int index,
                                    uint32_t count);

std::vector<int32_t> GetAlignedDims(
    const NNAdapterOperandDimensionType& target_dimensions,
    const NNAdapterOperandDimensionType& reference_dimensions);

nvinfer1::Dims ConvertToNVDims(const NNAdapterOperandDimensionType& dimensions,
                               bool ignore_batch = true);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
