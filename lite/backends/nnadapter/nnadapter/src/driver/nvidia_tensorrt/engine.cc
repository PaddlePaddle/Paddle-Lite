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

#include "driver/nvidia_tensorrt/engine.h"
#include <algorithm>
#include <unordered_set>
#include "driver/nvidia_tensorrt/optimizer/replace_softmax.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

static std::unordered_set<NNAdapterOperationType>
GetTensorrtSupportedOperationTypes() {
  std::unordered_set<NNAdapterOperationType> tensorrt_operations;
#define REGISTER_CONVERTER(__op_type__, __kernel_name__) \
  tensorrt_operations.insert(NNADAPTER_##__op_type__);
#include "driver/nvidia_tensorrt/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_NVIDIA_TENSORRT_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
  return tensorrt_operations;
}

static std::unordered_set<NNAdapterOperationType>
GetCudaSupportedOperationTypes() {
  std::unordered_set<NNAdapterOperationType> cuda_operations;
#define REGISTER_KERNEL(__op_type__, __kernel_name__) \
  cuda_operations.insert(NNADAPTER_##__op_type__);
#include "driver/nvidia_tensorrt/kernel/cuda/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_NVIDIA_TENSORRT_KERNELS_CUDA_ALL_H__
#undef REGISTER_KERNEL
  return cuda_operations;
}

static std::unordered_set<NNAdapterOperationType>
GetHostSupportedOperationTypes() {
  std::unordered_set<NNAdapterOperationType> host_operations;
#define REGISTER_KERNEL(__op_type__, __kernel_name__) \
  host_operations.insert(NNADAPTER_##__op_type__);
#include "driver/nvidia_tensorrt/kernel/host/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_NVIDIA_TENSORRT_KERNELS_HOST_ALL_H__
#undef REGISTER_KERNEL
  return host_operations;
}

void Program::Clear() {
  if (is_sub_model_from_cache_) {
    for (size_t i = 0; i < sub_models_.size(); i++) {
      auto& model = std::get<0>(sub_models_.at(i).second);
      if (model) {
        ClearModel(model);
        delete model;
        model = nullptr;
      }
    }
  }
  sub_models_.clear();
  sub_caches_.clear();
  sub_programs_.clear();
  input_tensors_.clear();
  temporary_tensors_.clear();
  output_tensors_.clear();
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(BuildFromModel(model), NNADAPTER_NO_ERROR);
  } else {
    NNADAPTER_CHECK_EQ(BuildFromCache(cache), NNADAPTER_NO_ERROR);
  }
  // Create sub_model
  for (size_t i = 0; i < sub_models_.size(); i++) {
    auto& sub_model = sub_models_.at(i);
    switch (static_cast<DeviceType>(sub_model.first)) {
      case kTensorrt:
        sub_programs_.emplace_back(new TensorrtProgram(
            context_, std::get<0>(sub_model.second), &sub_caches_.at(i)));
        break;
      case kCUDA:
        sub_programs_.emplace_back(new CudaProgram(
            context_, std::get<0>(sub_model.second), &sub_caches_.at(i)));
        break;
      case kHost:
        sub_programs_.emplace_back(new HostProgram(
            context_, std::get<0>(sub_model.second), &sub_caches_.at(i)));
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Not support device id: " << sub_model.first;
        break;
    }
  }
  // Build sub_model
  for (auto& sub_program : sub_programs_) {
    NNADAPTER_CHECK_EQ(sub_program->Build(), NNADAPTER_NO_ERROR);
  }
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(SerializeToCache(&cache->buffer), NNADAPTER_NO_ERROR);
  }
  for (auto& type : input_types_) {
    ConvertDynamicDimensions(&type.dimensions);
    max_batch_size_ =
        std::max(max_batch_size_, GetMaxBatchSize(type.dimensions));
  }
  for (auto& type : output_types_) {
    ConvertDynamicDimensions(&type.dimensions);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromModel(core::Model* model) {
  // Convert nnadapter standard ops to custom ops
  // ReplaceSoftmaxWithNaiveSoftmax(model);
  // ReplaceSoftmaxWithSpecialSoftmax(model);
  // Prepare input/output types
  for (auto& operand : model->input_operands) {
    input_types_.push_back(operand->type);
  }
  for (auto& operand : model->output_operands) {
    output_types_.push_back(operand->type);
  }
  // Partition model
  std::unordered_set<core::Operation*> tensorrt_operations;
  std::unordered_set<core::Operation*> cuda_operations;
  std::unordered_set<core::Operation*> host_operations;
  auto tensorrt_supported_operation_types =
      GetTensorrtSupportedOperationTypes();
  auto cuda_supported_operation_types = GetCudaSupportedOperationTypes();
  auto host_supported_operation_types = GetHostSupportedOperationTypes();
  for (auto& operation : model->operations) {
    if (tensorrt_supported_operation_types.count(operation.type)) {
      tensorrt_operations.insert(&operation);
    } else if (cuda_supported_operation_types.count(operation.type)) {
      cuda_operations.insert(&operation);
    } else if (host_supported_operation_types.count(operation.type)) {
      host_operations.insert(&operation);
    } else {
      NNADAPTER_LOG(FATAL) << "Not support operation type: " << operation.type;
    }
  }
  std::vector<std::pair<int, std::unordered_set<core::Operation*>>>
      supported_operations{{static_cast<int>(kTensorrt), tensorrt_operations},
                           {static_cast<int>(kCUDA), cuda_operations},
                           {static_cast<int>(kHost), host_operations}};
  PartitionModelIntoSubmodels(model, supported_operations, &sub_models_);
  NNADAPTER_CHECK(!sub_models_.empty());
  sub_caches_.resize(sub_models_.size());
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  DeserializeFromCache(&cache->buffer);
  input_types_ = cache->input_types;
  output_types_ = cache->output_types;
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get actual type
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNAdapterOperandType type;
    arg->access(arg->memory, &type, nullptr);
    // Check dimensions count
    uint32_t count = type.dimensions.count;
    auto& src_dimensions = input_types_.at(i).dimensions;
    if (count != src_dimensions.count) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
    // Check dimensions data
    bool is_explicit_dims = true;
    int32_t* data = type.dimensions.data;
    int32_t* src_data = src_dimensions.data;
    if (data[0] > max_batch_size_) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
    for (uint32_t j = 1; j < count; j++) {
      if (src_data[i] == NNADAPTER_UNKNOWN) {
        is_explicit_dims = false;
        break;
      }
      if (data[j] != src_data[j]) {
        return NNADAPTER_INVALID_DIMENSIONS;
      }
    }
    if (is_explicit_dims) continue;
    // Check dynamic dymensions data
    NNADAPTER_CHECK_EQ(src_dimensions.dynamic_count, 3U);
    for (uint32_t j = 1; j < count; j++) {
      if (data[j] < src_dimensions.dynamic_data[1][j] ||
          data[j] > src_dimensions.dynamic_data[2][j]) {
        return NNADAPTER_INVALID_DIMENSIONS;
      }
    }
  }
  return NNADAPTER_NO_ERROR;
}

static void SetTensor(Tensor* tensor,
                      void* data_ptr,
                      const NNAdapterOperandType& type,
                      bool is_device_buffer,
                      cudaStream_t stream) {
  auto dims = type.dimensions;
  std::vector<int32_t> shape(dims.data, dims.data + dims.count);
  auto data_type = ConvertToNVDataType(type.precision);
  if (is_device_buffer) {
    tensor->SetData(data_ptr, shape, data_type);
  } else {
    tensor->Resize(shape);
    tensor->SetDataType(data_type);
    uint32_t length =
        tensor->Length() * GetOperandPrecisionDataLength(type.precision);
    NNADAPTER_CHECK_EQ(
        cudaMemcpyAsync(
            tensor->Data(), data_ptr, length, cudaMemcpyHostToDevice, stream),
        cudaSuccess);
  }
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  // 1. Feed inputs
  cudaStream_t stream = context_->CudaStream();
  for (size_t i = 0; i < input_types_.size(); i++) {
    // Get input info
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    auto type = input_types_.at(i);
    bool is_device_buffer = false;
    auto data_ptr = arg->access(arg->memory, &type, &is_device_buffer);
    io_use_device_buffer_ |= is_device_buffer;
    NNADAPTER_CHECK(data_ptr);
    // Fill input tensor
    int index = -static_cast<int>(i) - 1;
    if (!input_tensors_.count(index)) {
      input_tensors_[index] = std::shared_ptr<Tensor>(new Tensor());
    }
    SetTensor(
        input_tensors_[index].get(), data_ptr, type, is_device_buffer, stream);
  }
  // 2. Execute sub_programs_ in order
  for (size_t i = 0; i < sub_programs_.size(); i++) {
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    std::vector<std::shared_ptr<Tensor>> output_tensors;
    auto input_indexes = std::get<2>(sub_models_.at(i).second);
    auto output_indexes = std::get<3>(sub_models_.at(i).second);
    // Find inputs
    for (auto input_index : input_indexes) {
      if (input_index < 0) {
        NNADAPTER_CHECK(input_tensors_.count(input_index));
        input_tensors.push_back(input_tensors_.at(input_index));
      } else {
        NNADAPTER_CHECK(temporary_tensors_.count(input_index));
        input_tensors.push_back(temporary_tensors_.at(input_index));
      }
    }
    // Init outputs
    for (auto output_index : output_indexes) {
      if (output_index < 0) {
        if (!output_tensors_.count(output_index)) {
          output_tensors_[output_index] = std::shared_ptr<Tensor>(new Tensor());
        }
        output_tensors.push_back(output_tensors_[output_index]);
      } else {
        if (!temporary_tensors_.count(output_index)) {
          temporary_tensors_[output_index] =
              std::shared_ptr<Tensor>(new Tensor());
        }
        output_tensors.push_back(temporary_tensors_[output_index]);
      }
    }
    sub_programs_[i]->Execute(&input_tensors, &output_tensors, stream);
  }
  // 3. Fetch outputs
  for (size_t i = 0; i < output_types_.size(); i++) {
    auto arg = FindArgumentByIndex(output_arguments, i, output_count);
    NNADAPTER_CHECK(arg) << "Output argument " << i << " does not exist!";
    int index = -static_cast<int>(i) - 1;
    auto dims = output_tensors_.at(index)->Dims();
    NNAdapterOperandType type = output_types_.at(i);
    type.dimensions.count = dims.size();
    memcpy(type.dimensions.data, dims.data(), dims.size() * sizeof(int32_t));
    auto output_tensor = output_tensors_.at(index);
    auto length = GetOperandTypeBufferLength(type);
    if (output_tensor->Data() && io_use_device_buffer_) {
      arg->access(arg->memory, &type, output_tensor->Data());
    } else if (output_tensor->Data()) {
      auto host_ptr = arg->access(arg->memory, &type, nullptr);
      NNADAPTER_CHECK_EQ(cudaMemcpyAsync(host_ptr,
                                         output_tensor->Data(),
                                         length,
                                         cudaMemcpyDeviceToHost,
                                         stream),
                         cudaSuccess);
    } else {
      auto host_ptr = arg->access(arg->memory, &type, nullptr);
      memcpy(host_ptr, output_tensor->Data(false), length);
    }
  }
  NNADAPTER_CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  return NNADAPTER_NO_ERROR;
}

int Program::SerializeToCache(std::vector<uint8_t>* buffer) {
  size_t size = sizeof(size_t);
  std::vector<std::vector<uint8_t>> model_buffers(sub_models_.size());
  for (size_t i = 0; i < sub_models_.size(); i++) {
    SerializeModel(std::get<0>(sub_models_.at(i).second), &model_buffers.at(i));
    size += sizeof(int32_t) + sizeof(bool) + sizeof(size_t) * 4 +
            model_buffers.at(i).size() +
            std::get<2>(sub_models_.at(i).second).size() * sizeof(int32_t) +
            std::get<3>(sub_models_.at(i).second).size() * sizeof(int32_t) +
            sub_caches_.at(i).size();
  }
  buffer->resize(size);
  void* ptr = reinterpret_cast<void*>(buffer->data());
  Serialize(&ptr, sub_models_.size());
  for (size_t i = 0; i < sub_models_.size(); i++) {
    Serialize(&ptr, sub_models_.at(i).first);
    Serialize(&ptr, model_buffers.at(i));
    Serialize(&ptr, std::get<1>(sub_models_.at(i).second));
    Serialize(&ptr, std::get<2>(sub_models_.at(i).second));
    Serialize(&ptr, std::get<3>(sub_models_.at(i).second));
    Serialize(&ptr, sub_caches_.at(i));
  }
  return NNADAPTER_NO_ERROR;
}

int Program::DeserializeFromCache(std::vector<uint8_t>* buffer) {
  const void* ptr = reinterpret_cast<void*>(buffer->data());
  size_t buffer_size = buffer->size();
  size_t sub_model_size;
  Deserialize(&ptr, &buffer_size, &sub_model_size);
  for (size_t i = 0; i < sub_model_size; i++) {
    int device_id;
    std::vector<uint8_t> model_buffer;
    // How to delete?
    core::Model* model;
    bool bool_value;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;
    std::vector<uint8_t> sub_cache;
    Deserialize(&ptr, &buffer_size, &device_id);
    Deserialize(&ptr, &buffer_size, &model_buffer);
    NNADAPTER_CHECK(
        DeserializeModel(model_buffer.data(), model_buffer.size(), &model));
    Deserialize(&ptr, &buffer_size, &bool_value);
    Deserialize(&ptr, &buffer_size, &input_indexes);
    Deserialize(&ptr, &buffer_size, &output_indexes);
    Deserialize(&ptr, &buffer_size, &sub_cache);
    sub_models_.emplace_back(
        device_id,
        std::make_tuple(model, bool_value, input_indexes, output_indexes));
    sub_caches_.emplace_back(sub_cache);
  }
  is_sub_model_from_cache_ = true;
  NNADAPTER_CHECK_EQ(buffer_size, 0UL);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
