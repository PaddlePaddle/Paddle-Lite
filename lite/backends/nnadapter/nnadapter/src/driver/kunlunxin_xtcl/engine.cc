// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/kunlunxin_xtcl/engine.h"
#include <utility>
#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
#include "optimizer/fuse_conv2d_add_into_conv2d.h"
#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  // KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS
  std::string selected_device_ids;
  if (key_values.count(KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS)) {
    selected_device_ids = key_values[KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS];
  } else {
    selected_device_ids = GetStringFromEnv(KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS);
  }
  if (!selected_device_ids.empty()) {
    selected_device_ids_ = string_split<int>(selected_device_ids, ",");
  } else {
    selected_device_ids_ = std::vector<int>({0});
  }
  NNADAPTER_CHECK_GE(selected_device_ids_.size(), 1);
  // Only supports specifying one device
  if (selected_device_ids_.size() > 1) {
    NNADAPTER_LOG(WARNING) << "Only supports specifying one device, so the "
                              "first one is selected and others will be "
                              "ignored.";
    auto first_device_id = selected_device_ids_[0];
    selected_device_ids_.clear();
    selected_device_ids_.push_back(first_device_id);
  }
  NNADAPTER_LOG(INFO) << "selected device ids: ";
  for (auto& selected_device_id : selected_device_ids_) {
    NNADAPTER_LOG(INFO) << selected_device_id;
  }
  // KUNLUNXIN_XTCL_DEVICE_TARGET
  if (key_values.count(KUNLUNXIN_XTCL_DEVICE_TARGET)) {
    device_target_ = key_values[KUNLUNXIN_XTCL_DEVICE_TARGET];
  } else {
    device_target_ = GetStringFromEnv(KUNLUNXIN_XTCL_DEVICE_TARGET);
  }
  NNADAPTER_LOG(INFO) << "device target: " << device_target_;
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  exprs_.clear();
  params_.clear();
  runtime_ = nullptr;
  input_tensors_.clear();
  output_tensors_.clear();
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (!cache->buffer.empty()) {
    // Build from cache
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    input_types_ = cache->input_types;
    auto output_count = cache->output_types.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_types_ = cache->output_types;
    // Create a runtime instance from a buffer
    runtime_ = LoadInstanceRuntimeFromBuffer(context_->first_device_id(),
                                             &cache->buffer);
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    FuseConv2DBatchNormIntoConv2D(model);
    FuseConv2DAddIntoConv2D(model);
    FuseConv2DActivationIntoConv2D(model);
    FuseMatMulAddIntoFullyConnected(model);
    FuseReshapeTransposeReshapeIntoChannelShuffle(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
    // Convert a NNAdapter model to a XTCL network
    Converter converter(&builder_, &params_, &exprs_);
    NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
    // Identify the inputs and outputs
    auto input_count = model->input_operands.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    if (input_count > 0) {
      input_types_.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        NNADAPTER_CHECK(exprs_.find(operand) != exprs_.end());
        input_types_[i] = operand->type;
      }
    }
    auto output_count = model->output_operands.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    xtcl::Array<xtcl::xExpr> output_exprs;
    output_types_.resize(output_count);
    for (size_t i = 0; i < output_count; i++) {
      auto operand = model->output_operands[i];
      NNADAPTER_CHECK(exprs_.find(operand) != exprs_.end());
      output_exprs.push_back(exprs_[operand].back());
      output_types_[i] = operand->type;
    }
    // Build a XTCL network to a runtime instance, and serialize it into a
    // buffer
    runtime_ = BuildInstanceRuntimeToBuffer(
        context_->first_device_id(),
        context_->device_target(),
        &builder_,
        &params_,
        &output_exprs,
        cache->token && cache->dir ? &cache->buffer : nullptr);
  }
  NNADAPTER_CHECK(runtime_ != nullptr) << "Invalid XTCL runtime instance!";
  // Initialize the model input and output DLTensors
  auto input_count = input_types_.size();
  for (size_t i = 0; i < input_count; i++) {
    const auto& type = input_types_[i];
    DLTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.device.device_type = kDLCPU;
    tensor.ndim = type.dimensions.count;
    tensor.dtype = ConvertToDLDataType(type.precision);
    input_tensors_.push_back(tensor);
  }
  auto output_count = output_types_.size();
  for (size_t i = 0; i < output_count; i++) {
    const auto& type = output_types_[i];
    DLTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.device.device_type = kDLCPU;
    tensor.ndim = type.dimensions.count;
    tensor.dtype = ConvertToDLDataType(type.precision);
    output_tensors_.push_back(tensor);
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
  std::vector<std::vector<int64_t>> input_shapes(input_count);
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = input_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    // Re-initialize the input tensors when the dimensions of inputs are changed
    // if dynamic shape is supported
    input_shapes[arg.index] = std::vector<int64_t>(
        type.dimensions.data, type.dimensions.data + type.dimensions.count);
    auto tensor = &input_tensors_[arg.index];
    tensor->data = buffer;
    tensor->shape = input_shapes[arg.index].data();
    runtime_->SetInput(runtime_->GetInputName(arg.index), tensor);
  }
  auto start_time = GetCurrentUS();
  runtime_->Run();
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  std::vector<std::vector<int64_t>> output_shapes(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from XTCL according
    // to the dynamic dimensions of inputs, fill them to 'type' and call the
    // 'access' function to re-allocate the host output memory
    output_shapes[arg.index] = std::vector<int64_t>(
        type->dimensions.data, type->dimensions.data + type->dimensions.count);
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto tensor = &output_tensors_[arg.index];
    tensor->data = buffer;
    tensor->shape = output_shapes[arg.index].data();
    runtime_->CopyOutputTo(arg.index, &output_tensors_[arg.index]);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
