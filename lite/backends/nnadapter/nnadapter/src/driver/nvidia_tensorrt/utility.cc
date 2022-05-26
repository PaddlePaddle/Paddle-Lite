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
#include <algorithm>

namespace nnadapter {
namespace nvidia_tensorrt {

Tensor::~Tensor() {
  if (own_cuda_buffer_ && cuda_buffer_) {
    cudaFree(cuda_buffer_);
  }
}

void Tensor::SetData(void* cuda_buffer,
                     const std::vector<int32_t>& dims,
                     nvinfer1::DataType data_type) {
  NNADAPTER_CHECK(cuda_buffer);
  NNADAPTER_CHECK(!dims.empty());
  cuda_buffer_ = cuda_buffer;
  dims_ = dims;
  data_type_ = data_type;
  cuda_buffer_length_ = Length() * GetNVTypeSize(data_type_);
  own_cuda_buffer_ = false;
}

void* Tensor::Data(bool return_cuda_buffer) {
  uint32_t dst_length = Length() * GetNVTypeSize(data_type_);
  if (return_cuda_buffer) {
    NNADAPTER_CHECK(!(host_buffer_length_ > 0 && cuda_buffer_length_ == 0))
        << "Host tensor should not return cuda buffer";
    if (dst_length > cuda_buffer_length_) {
      NNADAPTER_CHECK(own_cuda_buffer_)
          << "Should not reset external device buffer.";
      if (cuda_buffer_) {
        cudaFree(cuda_buffer_);
      }
      NNADAPTER_CHECK_EQ(cudaMalloc(&cuda_buffer_, dst_length), cudaSuccess);
      cuda_buffer_length_ = dst_length;
    }
    return cuda_buffer_;
  } else {
    if (dst_length > host_buffer_length_) {
      void* data = malloc(dst_length);
      host_buffer_.reset(data);
      host_buffer_length_ = dst_length;
    }
    if (cuda_buffer_length_ > 0) {
      NNADAPTER_CHECK_GE(host_buffer_length_, cuda_buffer_length_);
      NNADAPTER_CHECK_EQ(cudaMemcpy(host_buffer_.get(),
                                    cuda_buffer_,
                                    cuda_buffer_length_,
                                    cudaMemcpyDeviceToHost),
                         cudaSuccess);
    }
    return host_buffer_.get();
  }
}

uint32_t Tensor::Length() {
  if (dims_.empty()) return 0;
  uint32_t length = 1;
  for (auto i : dims_) {
    length *= static_cast<uint32_t>(i);
  }
  return length;
}

void TrtLogger::log(nvinfer1::ILogger::Severity severity,
                    const char* msg) TRT_NOEXCEPT {
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
template void Serialize(void** buffer, const size_t value);
template void Serialize(void** buffer, const bool value);
template void Serialize(void** buffer, const float value);

template <typename T>
void Serialize(void** buffer, const std::vector<T>& value) {
  Serialize(buffer, value.size());
  size_t nbyte = value.size() * sizeof(T);
  std::memcpy(*buffer, value.data(), nbyte);
  reinterpret_cast<char*&>(*buffer) += nbyte;
}
template void Serialize(void** buffer, const std::vector<uint8_t>& value);
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
                          size_t* value);
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
                          std::vector<uint8_t>* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          std::vector<int32_t>* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          std::vector<int64_t>* value);
template void Deserialize(const void** buffer,
                          size_t* buffer_size,
                          std::vector<float>* value);

void ConvertDynamicDimensions(NNAdapterOperandDimensionType* dimensions) {
  if (dimensions->dynamic_count == 0) return;
  int count = dimensions->count;
  int dynamic_count = dimensions->dynamic_count;
  auto& dynamic_data = dimensions->dynamic_data;
  std::vector<int32_t> opt_shape(dynamic_data[0], dynamic_data[0] + count);
  std::vector<int32_t> min_shape(opt_shape);
  std::vector<int32_t> max_shape(opt_shape);
  for (int i = 1; i < dynamic_count; i++) {
    for (int j = 0; j < count; j++) {
      if (dynamic_data[i][j] < min_shape[j]) {
        min_shape[j] = dynamic_data[i][j];
      }
      if (dynamic_data[i][j] > max_shape[j]) {
        max_shape[j] = dynamic_data[i][j];
      }
    }
  }
  memcpy(dynamic_data[0], min_shape.data(), sizeof(int32_t) * count);
  memcpy(dynamic_data[1], opt_shape.data(), sizeof(int32_t) * count);
  memcpy(dynamic_data[2], max_shape.data(), sizeof(int32_t) * count);
  dimensions->dynamic_count = 3;
}

int GetMaxBatchSize(const NNAdapterOperandDimensionType& dimensions) {
  int max_batch_size = std::max(1, dimensions.data[0]);
  for (int i = 0; i < dimensions.dynamic_count; i++) {
    max_batch_size = std::max(max_batch_size, dimensions.dynamic_data[i][0]);
  }
  return max_batch_size;
}

core::Argument* FindArgumentByIndex(core::Argument* arguments,
                                    int index,
                                    uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    if (arguments[i].index == index) {
      return &arguments[i];
    }
  }
  return static_cast<core::Argument*>(nullptr);
}

std::vector<int32_t> GetAlignedDims(
    const NNAdapterOperandDimensionType& target_dimensions,
    const NNAdapterOperandDimensionType& reference_dimensions) {
  auto dims0_count = target_dimensions.count;
  auto dims1_count = reference_dimensions.count;
  auto dims0_data = target_dimensions.data;
  std::vector<int32_t> dims(dims0_data, dims0_data + dims0_count);
  for (size_t i = 0; i < dims.size(); i++) {
    if (dims[i] == NNADAPTER_UNKNOWN) {
      dims[i] = -1;
    }
  }
  if (dims0_count < dims1_count) {
    dims.insert(dims.begin(), dims1_count - dims0_count, 1);
  }
  return dims;
}

nvinfer1::Dims ConvertToNVDims(const NNAdapterOperandDimensionType& dimensions,
                               bool ignore_batch) {
  int count = dimensions.count;
  auto data = dimensions.data;
  for (int i = 1; i < count; i++) {
    NNADAPTER_CHECK_NE(data[i], NNADAPTER_UNKNOWN);
  }
  nvinfer1::Dims dims;
  if (ignore_batch) {
    dims.nbDims = count - 1;
    memcpy(dims.d, data + 1, dims.nbDims * sizeof(int32_t));
  } else {
    NNADAPTER_CHECK_NE(data[0], NNADAPTER_UNKNOWN);
    dims.nbDims = count;
    memcpy(dims.d, data, dims.nbDims * sizeof(int32_t));
  }
  return dims;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
