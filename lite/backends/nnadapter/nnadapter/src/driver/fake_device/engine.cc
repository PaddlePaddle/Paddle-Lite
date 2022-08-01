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

#include "engine.h"  // NOLINT
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "converter/converter.h"
#include "converter/validator.h"
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
namespace fake_device {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from fake_ddk
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
}

int Program::Validate(const core::Model* model, bool* supported_operations) {
  Validator validator(context_);
  return validator.Apply(model, supported_operations);
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  return cache->buffer.empty() ? BuildFromModel(model, cache)
                               : BuildFromCache(cache);
}

int Program::BuildFromModel(core::Model* model, core::Cache* cache) {
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseConv2DBatchNormIntoConv2D(model);
  FuseConv2DAddIntoConv2D(model);
  FuseConv2DActivationIntoConv2D(model);
  FuseMatMulAddIntoFullyConnected(model);
  FuseReshapeTransposeReshapeIntoChannelShuffle(model);
  ConvertQuantizationSymmToAsymm(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a fake device graph
  graph_ = std::make_shared<fake_ddk::Graph>();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  Converter converter(graph_.get(), &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<fake_ddk::Tensor*> input_tensors;
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
  std::vector<fake_ddk::Tensor*> output_tensors(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    NNADAPTER_CHECK(output_tensors[i]);
    output_types_[i] = type;
  }
  graph_->IdentifyInputsAndOutputs(input_tensors, output_tensors);
  // Build the graph and create an execution for inference.
  execution_ = std::make_shared<fake_ddk::Execution>(graph_.get());
  if (cache->token && cache->dir) {
    if (execution_->Build(&cache->buffer) == fake_ddk::StatusType::SUCCESS) {
      NNADAPTER_VLOG(3) << "Serialize the model into a buffer success.";
    } else {
      NNADAPTER_LOG(FATAL) << "Failed to serialize the model into a buffer!";
    }
  } else {
    execution_->Build();
  }
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  auto input_count = cache->input_types.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  input_types_ = cache->input_types;
  auto output_count = cache->output_types.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  output_types_ = cache->output_types;
  // Deserialize a graph from a buffer
  graph_ = std::make_shared<fake_ddk::Graph>(cache->buffer);
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  NNADAPTER_VLOG(3) << "Deserialize a graph from a buffer success.";
  std::vector<fake_ddk::TensorAttr> input_attrs, output_attrs;
  NNADAPTER_CHECK(graph_->QueryInputsAndOutputs(&input_attrs, &output_attrs) ==
                  fake_ddk::StatusType::SUCCESS);
  NNADAPTER_CHECK_EQ(input_count, input_attrs.size());
  NNADAPTER_CHECK_EQ(output_count, output_attrs.size());
  // TODO(hong19860320) Check the precision, layout and shape.
  // Build the graph and create an execution for inference.
  execution_ = std::make_shared<fake_ddk::Execution>(graph_.get());
  execution_->Build();
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get the new dimensions
    auto& arg = input_arguments[i];
    NNAdapterOperandType new_type;
    arg.access(arg.memory, &new_type, nullptr);
    // Check whether the count and data of dimensions have been changed
    const NNAdapterOperandType& old_type = input_types_[arg.index];
    bool matched = MatchDimensions(new_type.dimensions.data,
                                   new_type.dimensions.count,
                                   old_type.dimensions.data,
                                   old_type.dimensions.count);
    if (!matched) {
      return NNADAPTER_INVALID_DIMENSIONS;
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
  // Prepare input tensors
  std::vector<fake_ddk::Argument> input_tensors;
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = input_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(type);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(buffer));
    }
    fake_ddk::Argument tensor;
    tensor.index = arg.index;
    // Update the shape of the input tensor with the new dimensions
    tensor.shape = ConvertToFakeDeviceDimensions(type.dimensions.data,
                                                 type.dimensions.count);
    tensor.buffer = buffer;
    input_tensors.push_back(tensor);
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK_EQ(execution_->SetInputs(input_tensors),
                     fake_ddk::StatusType::SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->Run(), fake_ddk::StatusType::SUCCESS);
  std::vector<fake_ddk::Argument> output_tensors;
  NNADAPTER_CHECK_EQ(execution_->GetOutputs(&output_tensors),
                     fake_ddk::StatusType::SUCCESS);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  NNADAPTER_CHECK_EQ(output_tensors.size(), output_count);
  // Read the output tensors
  auto FindArgumentByIndex = [&](
      core::Argument* arguments, int index, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
      if (arguments[i].index == index) {
        return &arguments[i];
      }
    }
    NNADAPTER_LOG(FATAL) << "Unable to find a argument with index=" << index
                         << " !";
    return static_cast<core::Argument*>(nullptr);
  };
  for (uint32_t i = 0; i < output_count; i++) {
    auto& tensor = output_tensors[i];
    NNADAPTER_CHECK_GE(tensor.index, 0);
    NNADAPTER_CHECK_LT(tensor.index, output_count);
    auto arg =
        FindArgumentByIndex(output_arguments, tensor.index, output_count);
    NNADAPTER_CHECK(arg->memory);
    NNADAPTER_CHECK(arg->access);
    auto type = output_types_[arg->index];
    NNADAPTER_CHECK_EQ(type.dimensions.count, tensor.shape.size());
    memcpy(type.dimensions.data,
           tensor.shape.data(),
           sizeof(int32_t) * tensor.shape.size());
    auto buffer = arg->access(arg->memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(type);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(tensor.buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    } else {
      memcpy(buffer, tensor.buffer, length);
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace fake_device
}  // namespace nnadapter
