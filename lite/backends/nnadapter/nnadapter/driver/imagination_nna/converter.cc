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

#include "driver/imagination_nna/converter.h"
#include <unistd.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include "optimizer/symm2asymm.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from imgdnnn
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  input_info_.clear();
  output_info_.clear();
  input_types_.clear();
  output_types_.clear();
  for (size_t i = 0; i < input_memory_.size(); i++) {
    if (input_memory_[i].first) {
      imgdnn_mgr_.DestroyMemory(input_memory_[i].first);
    }
  }
  input_memory_.clear();
  for (size_t i = 0; i < output_memory_.size(); i++) {
    if (output_memory_[i].first) {
      imgdnn_mgr_.DestroyMemory(output_memory_[i].first);
    }
  }
  output_memory_.clear();
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  ConvertQuantizationSymmToAsymm(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertPool2D(operation);
        break;
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
        ConvertActivation(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<imgdnn_tensor> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_types_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
      input_tensors[i] = tensors_[operand].back();
      NNADAPTER_CHECK(input_tensors[i]);
      input_types_[i] = type;
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<imgdnn_tensor> output_tensors(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    NNADAPTER_CHECK(output_tensors[i]);
    output_types_[i] = type;
  }
  imgdnn_mgr_.CreateNetworkObject(input_tensors.size(),
                                  input_tensors.data(),
                                  output_tensors.size(),
                                  output_tensors.data());
  // Get the info of inputs and outputs, and check the count and buffer size of
  // inputs and outputs
  uint32_t num_inputs;
  imgdnn_mgr_.GetNetworkObjectInputs(
      std::numeric_limits<unsigned int>::max(), nullptr, &num_inputs);
  NNADAPTER_CHECK_EQ(input_count, num_inputs);
  if (input_count > 0) {
    input_info_.resize(input_count);
    input_memory_.resize(input_count,
                         std::pair<imgdnn_memory, size_t>(nullptr, 0));
    imgdnn_mgr_.GetNetworkObjectInputs(
        input_count, input_info_.data(), nullptr);
  }
  uint32_t num_outputs;
  imgdnn_mgr_.GetNetworkObjectOutputs(
      std::numeric_limits<unsigned int>::max(), nullptr, &num_outputs);
  NNADAPTER_CHECK_EQ(output_count, num_outputs);
  output_info_.resize(output_count);
  output_memory_.resize(output_count,
                        std::pair<imgdnn_memory, size_t>(nullptr, 0));
  imgdnn_mgr_.GetNetworkObjectOutputs(
      output_count, output_info_.data(), nullptr);
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  NNADAPTER_CHECK_EQ(input_types_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_types_.size(), output_count);
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &input_types_[arg.index];
    auto host_ptr = arg.access(arg.memory, type);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(*type);
    auto& input_memory = input_memory_[arg.index];
    if (!input_memory.first || input_memory.second < length) {
      if (input_memory.first) {
        imgdnn_mgr_.DestroyMemory(input_memory.first);
      }
      input_memory.first = imgdnn_mgr_.AllocateMemory(length);
      NNADAPTER_CHECK(input_memory.first);
      input_memory.second = length;
      imgdnn_mgr_.AddBindingInput(input_info_[arg.index], input_memory.first);
    }
    auto device_ptr = imgdnn_mgr_.LockMemory(input_memory.first,
                                             IMGDNN_LOCK_ACCESS_WRITE_ONLY);
    if (IsUInt8AsymmPerLayerQuantization(type->precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(host_ptr),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(device_ptr));
    } else {
      memcpy(host_ptr, device_ptr, length);
    }
    imgdnn_mgr_.UnlockMemory(input_memory.first);
  }
  std::vector<std::pair<size_t, std::pair<void*, imgdnn_memory>>>
      output_buffers(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from imgdnn
    // according to the dynamic dimensions of the inputs, fill them to 'type'
    // and call the 'access' function to re-allocate the host output memory
    auto host_ptr = arg.access(arg.memory, type);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(*type);
    auto& output_memory = output_memory_[arg.index];
    if (!output_memory.first || output_memory.second < length) {
      if (output_memory.first) {
        imgdnn_mgr_.DestroyMemory(output_memory.first);
      }
      output_memory.first = imgdnn_mgr_.AllocateMemory(length);
      NNADAPTER_CHECK(output_memory.first);
      output_memory.second = length;
      imgdnn_mgr_.AddBindingOutput(output_info_[arg.index],
                                   output_memory.first);
    }
    output_buffers[arg.index].first = length;
    output_buffers[arg.index].second.first = host_ptr;
    output_buffers[arg.index].second.second = output_memory.first;
  }
  auto start_time = GetCurrentUS();
  imgdnn_mgr_.ExecuteNetworkObject(true, 0, nullptr, nullptr);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto host_ptr = output_buffers[i].second.first;
    auto length = output_buffers[i].first;
    auto device_ptr = imgdnn_mgr_.LockMemory(output_buffers[i].second.second,
                                             IMGDNN_LOCK_ACCESS_READ_ONLY);
    if (IsUInt8AsymmPerLayerQuantization(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(device_ptr),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(host_ptr));
    } else {
      memcpy(device_ptr, host_ptr, length);
    }
    imgdnn_mgr_.UnlockMemory(output_buffers[i].second.second);
  }
  return NNADAPTER_NO_ERROR;
}

imgdnn_tensor Program::GetMappedTensor(hal::Operand* operand) {
  auto it = tensors_.find(operand);
  if (it != tensors_.end()) {
    return it->second.back();
  }
  return nullptr;
}

imgdnn_tensor Program::UpdateTensorMap(hal::Operand* operand,
                                       imgdnn_tensor tensor) {
  auto it = tensors_.find(operand);
  if (it == tensors_.end()) {
    auto result =
        tensors_.insert(std::make_pair(operand, std::vector<imgdnn_tensor>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

imgdnn_tensor Program::AddTensor(int32_t* dimensions,
                                 uint32_t dimension_count,
                                 imgdnn_type type,
                                 const float* quant_scales,
                                 const int32_t* zero_point,
                                 uint32_t quant_scale_count,
                                 uint32_t quant_channel_dim,
                                 void* buffer) {
  imgdnn_tensor tensor = nullptr;
  imgdnn_tensor_descriptor desc;
  desc.type = type;
  NNADAPTER_CHECK(dimensions);
  NNADAPTER_CHECK_GT(dimension_count, 0);
  ConvertDimensions(dimensions, dimension_count, desc.size, &desc.dimensions);
  if (quant_scales && quant_scale_count > 0) {
    // Quantization types
    if (quant_scale_count > 1) {
      // Symmetric and asymmetric per-channel quantization
      if (zero_point) {
        // Asymmetric per-channel quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_QPA_U8);
      } else {
        // Symmetric per-channel quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_QPA_I8);
      }
      desc.quant_param.per_axis = imgdnnCreatePerAxisQuantParam(
          quant_channel_dim, quant_scale_count, quant_scales, zero_point);
      NNADAPTER_CHECK(desc.quant_param.per_axis != nullptr);
      tensor = buffer ? imgdnn_mgr_.CreateFixedInputTensor(&desc, buffer, true)
                      : imgdnn_mgr_.CreateInputTensor(&desc);
      imgdnnDestroyPerAxisQuantParam(desc.quant_param.per_axis);
    } else {
      desc.quant_param.scale = quant_scales[0];
      if (zero_point) {
        // Asymmetric per-layer quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_Q_U8);
        desc.quant_param.zero_point = zero_point[0];
      } else {
        // Symmetric per-layer quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_Q_I8);
        // zeroPoint = 0
      }
      tensor = buffer ? imgdnn_mgr_.CreateFixedInputTensor(&desc, buffer, true)
                      : imgdnn_mgr_.CreateInputTensor(&desc);
    }
  } else {
    // TODO(hong19860320) Supports the normal types, such as float, int32 etc.
    NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_I32);
    NNADAPTER_CHECK(buffer);
    tensor = imgdnn_mgr_.CreateFixedInputTensor(&desc, buffer, true);
  }
  return tensor;
}

imgdnn_tensor Program::AddTensor(const NNAdapterOperandType* type,
                                 void* buffer,
                                 std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimension_count; i++) {
      dimensions.push_back(type->dimensions[i]);
    }
  }
  NNADAPTER_CHECK_EQ(type->layout, NNADAPTER_NCHW);
  const float* quant_scales = nullptr;
  const int32_t* zero_point = nullptr;
  uint32_t quant_scale_count = 0;
  uint32_t quant_channel_dim = 0;
  switch (type->precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      quant_scales = &type->asymm_per_layer_params.scale;
      zero_point = &type->asymm_per_layer_params.zero_point;
      quant_scale_count = 1;
      break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
      quant_scales = type->symm_per_channel_params.scales;
      quant_scale_count = type->symm_per_channel_params.scale_count;
      quant_channel_dim = type->symm_per_channel_params.channel_dim;
      break;
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL:
      // Only for bias
      NNADAPTER_CHECK(type->lifetime == NNADAPTER_CONSTANT_COPY ||
                      type->lifetime == NNADAPTER_CONSTANT_REFERENCE);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can not add a imgdnn_tensor with precision="
                           << OperandPrecisionCodeToString(type->precision)
                           << " !";
      break;
  }
  return AddTensor(dimensions.data(),
                   dimensions.size(),
                   ConvertPrecision(type->precision),
                   quant_scales,
                   zero_point,
                   quant_scale_count,
                   quant_channel_dim,
                   buffer);
}

imgdnn_tensor Program::AddQuant8ConstantTensor(uint8_t* values,
                                               int32_t* dimensions,
                                               uint32_t dimension_count,
                                               float quant_scale,
                                               int32_t zero_point) {
  return AddTensor(dimensions,
                   dimension_count,
                   IMGDNN_TYPE_Q_U8,
                   &quant_scale,
                   &zero_point,
                   1,
                   0,
                   values);
}

imgdnn_tensor Program::AddQuant8ConstantTensor(int8_t* values,
                                               int32_t* dimensions,
                                               uint32_t dimension_count,
                                               float* quant_scales,
                                               uint32_t quant_scale_count,
                                               uint32_t quant_channel_dim) {
  return AddTensor(dimensions,
                   dimension_count,
                   IMGDNN_TYPE_Q_I8,
                   quant_scales,
                   nullptr,
                   quant_scale_count,
                   quant_channel_dim,
                   values);
}

imgdnn_tensor Program::AddQuant32ConstantTensor(int32_t* values,
                                                int32_t* dimensions,
                                                uint32_t dimension_count,
                                                float quant_scale) {
  return AddTensor(dimensions,
                   dimension_count,
                   IMGDNN_TYPE_I32,
                   &quant_scale,
                   nullptr,
                   1,
                   0,
                   values);
}

imgdnn_tensor Program::ConvertOperand(hal::Operand* operand,
                                      std::vector<int32_t> dimensions) {
  if (IsConstantOperand(operand)) {
    auto constant_tensor =
        AddTensor(&operand->type, operand->buffer, dimensions);
    UpdateTensorMap(operand, constant_tensor);
    return constant_tensor;
  } else if (IsModelInputOperand(operand)) {
    auto input_tensor = AddTensor(&operand->type, nullptr, dimensions);
    UpdateTensorMap(operand, input_tensor);
    return input_tensor;
  } else {
    NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                            "converted to imgdnn_tensor!";
  }
  return nullptr;
}

}  // namespace imagination_nna
}  // namespace nnadapter
