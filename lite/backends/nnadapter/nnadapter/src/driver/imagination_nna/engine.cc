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

#include "driver/imagination_nna/engine.h"
#include <unistd.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include "driver/imagination_nna/converter/converter.h"
#include "optimizer/convert_quantization_symm_to_asymm.h"
#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
#include "optimizer/fuse_conv2d_add_into_conv2d.h"
#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
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
      imgdnn_mgr_->DestroyMemory(input_memory_[i].first);
    }
  }
  input_memory_.clear();
  for (size_t i = 0; i < output_memory_.size(); i++) {
    if (output_memory_[i].first) {
      imgdnn_mgr_->DestroyMemory(output_memory_[i].first);
    }
  }
  output_memory_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseConv2DBatchNormIntoConv2D(model);
  FuseConv2DAddIntoConv2D(model);
  FuseConv2DActivationIntoConv2D(model);
  FuseMatMulAddIntoFullyConnected(model);
  FuseReshapeTransposeReshapeIntoChannelShuffle(model);
  ConvertQuantizationSymmToAsymm(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a imgdnn network
  imgdnn_mgr_ = std::make_shared<ImgdnnManager>();
  if (!imgdnn_mgr_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  Converter converter(imgdnn_mgr_.get(), &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
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
      input_tensors[i] = tensors_[operand].front();
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
  imgdnn_mgr_->CreateNetworkObject(input_tensors.size(),
                                   input_tensors.data(),
                                   output_tensors.size(),
                                   output_tensors.data());
  // Get the info of inputs and outputs, and check the count and buffer size of
  // inputs and outputs
  uint32_t num_inputs;
  imgdnn_mgr_->GetNetworkObjectInputs(
      std::numeric_limits<unsigned int>::max(), nullptr, &num_inputs);
  NNADAPTER_CHECK_EQ(input_count, num_inputs);
  if (input_count > 0) {
    input_info_.resize(input_count);
    input_memory_.resize(input_count,
                         std::pair<imgdnn_memory, size_t>(nullptr, 0));
    imgdnn_mgr_->GetNetworkObjectInputs(
        input_count, input_info_.data(), nullptr);
  }
  uint32_t num_outputs;
  imgdnn_mgr_->GetNetworkObjectOutputs(
      std::numeric_limits<unsigned int>::max(), nullptr, &num_outputs);
  NNADAPTER_CHECK_EQ(output_count, num_outputs);
  output_info_.resize(output_count);
  output_memory_.resize(output_count,
                        std::pair<imgdnn_memory, size_t>(nullptr, 0));
  imgdnn_mgr_->GetNetworkObjectOutputs(
      output_count, output_info_.data(), nullptr);
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get actual type
    auto& arg = input_arguments[i];
    NNAdapterOperandType type;
    arg.access(arg.memory, &type, nullptr);
    // Check dimensions count
    uint32_t count = type.dimensions.count;
    int32_t* data = type.dimensions.data;
    auto& src_dimensions = input_types_[i].dimensions;
    int32_t* src_data = src_dimensions.data;
    if (count != src_dimensions.count) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
    // Check dimensions data
    for (uint32_t j = 0; j < count; j++) {
      if (data[j] != src_data[j]) {
        return NNADAPTER_INVALID_DIMENSIONS;
      }
    }
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  NNADAPTER_CHECK_EQ(input_types_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_types_.size(), output_count);
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = input_types_[arg.index];
    auto host_ptr = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(type);
    auto& input_memory = input_memory_[arg.index];
    if (!input_memory.first || input_memory.second < length) {
      if (input_memory.first) {
        imgdnn_mgr_->DestroyMemory(input_memory.first);
      }
      input_memory.first = imgdnn_mgr_->AllocateMemory(length);
      NNADAPTER_CHECK(input_memory.first);
      input_memory.second = length;
      imgdnn_mgr_->AddBindingInput(input_info_[arg.index], input_memory.first);
    }
    auto device_ptr = imgdnn_mgr_->LockMemory(input_memory.first,
                                              IMGDNN_LOCK_ACCESS_WRITE_ONLY);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(host_ptr),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(device_ptr));
    } else {
      memcpy(device_ptr, host_ptr, length);
    }
    imgdnn_mgr_->UnlockMemory(input_memory.first);
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
    auto host_ptr = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(*type);
    auto& output_memory = output_memory_[arg.index];
    if (!output_memory.first || output_memory.second < length) {
      if (output_memory.first) {
        imgdnn_mgr_->DestroyMemory(output_memory.first);
      }
      output_memory.first = imgdnn_mgr_->AllocateMemory(length);
      NNADAPTER_CHECK(output_memory.first);
      output_memory.second = length;
      imgdnn_mgr_->AddBindingOutput(output_info_[arg.index],
                                    output_memory.first);
    }
    output_buffers[arg.index].first = length;
    output_buffers[arg.index].second.first = host_ptr;
    output_buffers[arg.index].second.second = output_memory.first;
  }
  auto start_time = GetCurrentUS();
  imgdnn_mgr_->ExecuteNetworkObject(true, 0, nullptr, nullptr);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto host_ptr = output_buffers[i].second.first;
    auto length = output_buffers[i].first;
    auto device_ptr = imgdnn_mgr_->LockMemory(output_buffers[i].second.second,
                                              IMGDNN_LOCK_ACCESS_READ_ONLY);
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(device_ptr),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(host_ptr));
    } else {
      memcpy(host_ptr, device_ptr, length);
    }
    imgdnn_mgr_->UnlockMemory(output_buffers[i].second.second);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
