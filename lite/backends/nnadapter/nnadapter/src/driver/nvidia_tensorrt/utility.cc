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

#include "driver/nvidia_tensorrt/utility.h"
#include "utility/debug.h"

namespace nnadapter {
namespace nvidia_tensorrt {

void TrtLogger::log(nvinfer1::ILogger::Severity severity,
                    const char* msg) noexcept {
  switch (severity) {
    case Severity::kVERBOSE:
      NNADAPTER_VLOG(3) << "[Tensorrt]" << msg;
      break;
    case Severity::kINFO:
      NNADAPTER_VLOG(1) << "[Tensorrt]" << msg;
      break;
    case Severity::kWARNING:
      NNADAPTER_VLOG(2) << "[Tensorrt]" << msg;
      break;
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      NNADAPTER_LOG(ERROR) << "[Tensorrt]" << msg;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "[Tensorrt]"
                           << "Unpported severity level: "
                           << static_cast<int>(severity);
      break;
  }
}

nvinfer1::DataType ConvertToNVDataType(
    NNAdapterOperandPrecisionCode input_precision) {
  nvinfer1::DataType output_precision;
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = nvinfer1::DataType::kBOOL;
      break;
    case NNADAPTER_INT8:
    case NNADAPTER_UINT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = nvinfer1::DataType::kINT8;
      break;
    case NNADAPTER_INT32:
      output_precision = nvinfer1::DataType::kINT32;
      break;
    case NNADAPTER_FLOAT16:
      output_precision = nvinfer1::DataType::kHALF;
      break;
    case NNADAPTER_FLOAT32:
      output_precision = nvinfer1::DataType::kFLOAT;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to nvinfer1::DataType !";
      break;
  }
  return output_precision;
}

template <>
nvinfer1::DataType GetNVDateType<float>() {
  return nvinfer1::DataType::kFLOAT;
}

uint32_t GetNVTypeSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported data type: "
                           << static_cast<int>(type);
  }
  return 0;
}

template <typename T>
size_t SerializedSize(const T value) {
  return sizeof(T);
}
template size_t SerializedSize(const int32_t value);
template size_t SerializedSize(const int64_t value);
template size_t SerializedSize(const bool value);
template size_t SerializedSize(const float value);

template <typename T>
size_t SerializedSize(const std::vector<T>& value) {
  return sizeof(value.size()) + sizeof(T) * value.size();
}
template size_t SerializedSize(const std::vector<int32_t>& value);
template size_t SerializedSize(const std::vector<int64_t>& value);
template size_t SerializedSize(const std::vector<float>& value);

template <typename T>
void Serialize(void** buffer, const T value) {
  std::memcpy(*buffer, &value, sizeof(T));
  reinterpret_cast<char*&>(*buffer) += sizeof(T);
}
template void Serialize(void** buffer, const int32_t value);
template void Serialize(void** buffer, const int64_t value);
template void Serialize(void** buffer, const bool value);
template void Serialize(void** buffer, const float value);

template <typename T>
void Serialize(void** buffer, const std::vector<T>& value) {
  Serialize(buffer, value.size());
  size_t nbyte = value.size() * sizeof(T);
  std::memcpy(*buffer, value.data(), nbyte);
  reinterpret_cast<char*&>(*buffer) += nbyte;
}
template void Serialize(void** buffer, const std::vector<int32_t>& value);
template void Serialize(void** buffer, const std::vector<int64_t>& value);
template void Serialize(void** buffer, const std::vector<float>& value);

template <typename T>
void Deserialize(const void** buffer, size_t* buffer_size, T* value) {
  NNADAPTER_CHECK_GE(*buffer_size, sizeof(T));
  std::memcpy(value, *buffer, sizeof(T));
  reinterpret_cast<const char*&>(*buffer) += sizeof(T);
  *buffer_size -= sizeof(T);
}
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          int32_t* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          int64_t* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          bool* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          float* value);

template <typename T>
void Deserialize(const void** buffer,
                 size_t* buffer_size,
                 std::vector<T>* value) {
  NNADAPTER_CHECK_GE(*buffer_size, sizeof(value->size()));
  size_t size;
  Deserialize(buffer, buffer_size, &size);
  size_t nbyte = size * sizeof(T);
  NNADAPTER_CHECK_GE(*buffer_size, nbyte);
  value->resize(size);
  std::memcpy(value->data(), *buffer, nbyte);
  reinterpret_cast<const char*&>(*buffer) += nbyte;
  *buffer_size -= nbyte;
}
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          std::vector<int32_t>* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          std::vector<int64_t>* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          std::vector<float>* value);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
