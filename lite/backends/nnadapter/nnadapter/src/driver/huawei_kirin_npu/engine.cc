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

#include "driver/huawei_kirin_npu/engine.h"
#include <utility>
#include "driver/huawei_kirin_npu/optimizer/fix_multiple_outputs_ops.h"
#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
#include "optimizer/fuse_conv2d_add_into_conv2d.h"
#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from HiAI service
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  operators_.clear();
  model_name_ = "";
  model_client_ = nullptr;
  input_tensors_.clear();
  output_tensors_.clear();
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  std::vector<uint8_t> model_content;
  std::vector<uint8_t>* model_buffer = nullptr;
  if (!cache->buffer.empty()) {
    // Build from cache
    model_buffer = &cache->buffer;
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    input_types_ = cache->input_types;
    auto output_count = cache->output_types.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_types_ = cache->output_types;
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    FuseConv2DBatchNormIntoConv2D(model);
    FuseConv2DAddIntoConv2D(model);
    FuseConv2DActivationIntoConv2D(model);
    FuseReshapeTransposeReshapeIntoChannelShuffle(model);
    FixMultipleOutputsOps(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
    // Convert a NNAdapter model to a GE graph
    Converter converter(&operators_);
    NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
    // Identify the inputs and outputs
    auto input_count = model->input_operands.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    std::vector<ge::Operator> input_operators;
    if (input_count > 0) {
      input_operators.resize(input_count);
      input_types_.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
        input_operators[i] = *operators_[operand].front()->op();
        input_types_[i] = operand->type;
      }
    }
    auto output_count = model->output_operands.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    std::vector<ge::Operator> output_operators(output_count);
    output_types_.resize(output_count);
    for (size_t i = 0; i < output_count; i++) {
      auto operand = model->output_operands[i];
      NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
      output_operators[i] = *operators_[operand].back()->op();
      output_types_[i] = operand->type;
    }
    if (cache->token && cache->dir) {
      model_buffer = &cache->buffer;
    } else {
      model_buffer = &model_content;
    }
    // Build a GE graph to a HiAI OM model, and serialize it into a buffer
    if (!BuildOMModelToBuffer(
            input_operators, output_operators, model_buffer)) {
      NNADAPTER_LOG(FATAL)
          << "Failed to build a HiAI OM model and serialize it into a buffer!";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    } else {
      NNADAPTER_VLOG(3)
          << "Build a HiAI OM model and serialize it into a buffer success.";
    }
  }
  NNADAPTER_CHECK(model_buffer);
  // Load a HiAI OM model from a buffer, and create a HiAI model manager
  // client(from HiAI service) for inference
  bool model_comp = true;
  model_name_ = string_format("0x%X.om", cache);
  model_client_ = LoadOMModelFromBuffer(model_name_,
                                        model_buffer,
                                        &model_comp,
                                        context_->freq_level(),
                                        context_->framework_type(),
                                        context_->model_type(),
                                        context_->device_type());
  if (!model_client_) {
    NNADAPTER_LOG(FATAL) << "Failed to load a HiAI OM model from a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Initialize the HiAI input and output tensors
  std::vector<hiai::TensorDimension> input_dimensions, output_dimensions;
  if (model_client_->GetModelIOTensorDim(
          model_name_, input_dimensions, output_dimensions) !=
      hiai::AI_SUCCESS) {
    NNADAPTER_LOG(FATAL) << "Failed to call GetModelIOTensorDim to get the "
                            "dimensions of input and output tensors!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto input_count = input_types_.size();
  NNADAPTER_CHECK_EQ(input_dimensions.size(), input_count);
  if (input_count > 0) {
    input_tensors_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto n = input_dimensions[i].GetNumber();
      auto c = input_dimensions[i].GetChannel();
      auto h = input_dimensions[i].GetHeight();
      auto w = input_dimensions[i].GetWidth();
      NNADAPTER_VLOG(3) << "HiAI input tensors[" << i << "]: " << n << "," << c
                        << "," << h << "," << w;
      NNADAPTER_CHECK_EQ(
          ProductionOfDimensions(input_types_[i].dimensions.data,
                                 input_types_[i].dimensions.count),
          n * c * h * w);
      input_tensors_[i].reset(new hiai::AiTensor);
      input_tensors_[i]->Init(&(input_dimensions[i]));
    }
  }
  auto output_count = output_types_.size();
  NNADAPTER_CHECK_EQ(output_dimensions.size(), output_count);
  output_tensors_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto n = output_dimensions[i].GetNumber();
    auto c = output_dimensions[i].GetChannel();
    auto h = output_dimensions[i].GetHeight();
    auto w = output_dimensions[i].GetWidth();
    NNADAPTER_VLOG(3) << "HiAI output tensors[" << i << "]: " << n << "," << c
                      << "," << h << "," << w;
    NNADAPTER_CHECK_EQ(
        ProductionOfDimensions(output_types_[i].dimensions.data,
                               output_types_[i].dimensions.count),
        n * c * h * w);
    output_tensors_[i].reset(new hiai::AiTensor);
    output_tensors_[i]->Init(&(output_dimensions[i]));
  }
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
    auto buffer = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(type);
    // TODO(hong19860320) Re-initialize the input tensors when the dimensions of
    // inputs are changed if dynamic shape is supported
    NNADAPTER_CHECK_EQ(length, input_tensors_[arg.index]->GetSize());
    memcpy(input_tensors_[arg.index]->GetBuffer(), buffer, length);
  }
  std::string key = "model_name";  // Note: key seems must be model_name
  hiai::AiContext model_context;
  model_context.AddPara(key, model_name_);
  int istamp;
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK_EQ(
      model_client_->Process(
          model_context, input_tensors_, output_tensors_, 1000, istamp),
      hiai::AI_SUCCESS);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from HiAI according
    // to the dynamic dimensions of inputs, fill them to 'type' and call the
    // 'access' function to re-allocate the host output memory
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    NNADAPTER_CHECK_EQ(length, output_tensors_[arg.index]->GetSize());
    memcpy(buffer, output_tensors_[arg.index]->GetBuffer(), length);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
