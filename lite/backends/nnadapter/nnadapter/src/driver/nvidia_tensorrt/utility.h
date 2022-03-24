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
#include <memory>
#include <vector>
#include "core/types.h"
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

typedef enum {
  kFloat32 = 0,
  kFloat16 = 1,
  kInt8 = 2,
} PrecisionMode;

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override;

  static TrtLogger* Global() noexcept {
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

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
