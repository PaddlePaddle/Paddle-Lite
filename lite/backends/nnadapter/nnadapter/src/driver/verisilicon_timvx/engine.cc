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

#include "driver/verisilicon_timvx/engine.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "driver/verisilicon_timvx/converter/converter.h"
#include "driver/verisilicon_timvx/optimizer/convert_fill_like_into_mul_add.h"
#include "driver/verisilicon_timvx/optimizer/unpack_op_fusion.h"
#include "optimizer/constant_fold_operations.h"
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
namespace verisilicon_timvx {

Context::Context(void* device, const char* properties) : device_(device) {
  // By dafault, set the
  // TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION
  // as 1000, user can modify the threshold by context_property or ENV
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  if (key_values.count(
          TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION)) {
    batchnorm_fusion_max_allowed_quant_scale_deviation_ = string_parse<double>(
        key_values[TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION]);
  } else {
    batchnorm_fusion_max_allowed_quant_scale_deviation_ = GetDoubleFromEnv(
        TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION, 1000.f);
  }
  NNADAPTER_LOG(INFO) << "bn_fusion_max_allowed_quant_scale_deviation: "
                      << batchnorm_fusion_max_allowed_quant_scale_deviation_;
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  graph_ = nullptr;
  ctx_ = nullptr;
  input_types_.clear();
  output_types_.clear();
  input_tensors_.clear();
  output_tensors_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  // Prepare tim-vx context and graph
  ctx_ = tim::vx::Context::Create();
  if (!ctx_) {
    NNADAPTER_LOG(FATAL) << "Failed to create the tim-vx context!";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  graph_ = ctx_->CreateGraph();
  if (!graph_) {
    NNADAPTER_LOG(FATAL) << "Failed to create the tim-vx graph!";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  if (!cache->buffer.empty()) {
    // Build from cache
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    if (input_count > 0) {
      input_tensors_.resize(input_count);
      input_types_ = cache->input_types;
      for (size_t i = 0; i < input_count; i++) {
        const auto& type = cache->input_types[i];
        input_tensors_[i] = CreateTimVXTensor(graph_.get(), &type);
        NNADAPTER_CHECK(input_tensors_[i]);
      }
    }
    auto output_count = cache->output_types.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_tensors_.resize(input_count);
    output_types_ = cache->output_types;
    for (size_t i = 0; i < output_count; i++) {
      const auto& type = cache->output_types[i];
      output_tensors_[i] = CreateTimVXTensor(graph_.get(), &type);
      NNADAPTER_CHECK(output_tensors_[i]);
    }
    auto nbg_op = graph_->CreateOperation<tim::vx::ops::NBG>(
        reinterpret_cast<const char*>(cache->buffer.data()),
        input_count,
        output_count);
    NNADAPTER_CHECK(nbg_op != nullptr)
        << "Create NBG operation from cache graph success.";
    nbg_op->BindInputs(input_tensors_);
    nbg_op->BindOutputs(output_tensors_);
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    FuseConv2DBatchNormIntoConv2D(
        model, context_->batchnorm_fusion_max_allowed_quant_scale_deviation());
    FuseConv2DAddIntoConv2D(model);
    FuseConv2DBatchNormIntoConv2D(
        model, context_->batchnorm_fusion_max_allowed_quant_scale_deviation());
    FuseConv2DActivationIntoConv2D(model);
    FuseMatMulAddIntoFullyConnected(model);
    FuseReshapeTransposeReshapeIntoChannelShuffle(model);
    ConvertFillLikeIntoMulAdd(model);
    ConstantFoldOperations(model);
    UnpackOpFusion(model);
    ConvertQuantizationSymmToAsymm(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
    // Convert a NNAdapter model to a tim-vx graph
    Converter converter(graph_.get(), &tensors_);
    NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
    // Indentify the inputs and outputs
    auto input_count = model->input_operands.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    if (input_count > 0) {
      input_tensors_.resize(input_count);
      input_types_.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        const auto& type = operand->type;
        NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
        input_tensors_[i] = tensors_[operand].front();
        NNADAPTER_CHECK(input_tensors_[i]);
        input_types_[i] = type;
      }
    }
    auto output_count = model->output_operands.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_tensors_.resize(output_count);
    output_types_.resize(output_count);
    for (size_t i = 0; i < output_count; i++) {
      auto operand = model->output_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
      output_tensors_[i] = tensors_[operand].back();
      NNADAPTER_CHECK(output_tensors_[i]);
      output_types_[i] = type;
    }
    // Compile tim-vx graph and serialize to NBG(Network Binary Graph)
    if (cache->token && cache->dir) {
      size_t nbg_size = 0;
      if (!graph_->CompileToBinary(nullptr, &nbg_size)) {
        NNADAPTER_LOG(FATAL)
            << "Failed to compile tim-vx graph and get NBG size!";
        return NNADAPTER_DEVICE_INTERNAL_ERROR;
      }
      NNADAPTER_CHECK_GT(nbg_size, 0);
      cache->buffer.resize(nbg_size);
      if (!graph_->CompileToBinary(cache->buffer.data(), &nbg_size)) {
        NNADAPTER_LOG(FATAL)
            << "Failed to compile tim-vx graph and get NBG data!";
        return NNADAPTER_DEVICE_INTERNAL_ERROR;
      }
      NNADAPTER_VLOG(3) << "Build the tim-vx graph and get the NBG(Network "
                           "Binary Graph) success.";
      return NNADAPTER_NO_ERROR;
    }
  }
  if (!graph_->Compile()) {
    NNADAPTER_LOG(FATAL) << "Failed to compile tim-vx graph!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  NNADAPTER_VLOG(3) << "Build the tim-vx graph success.";
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
  NNADAPTER_CHECK_EQ(input_tensors_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_tensors_.size(), output_count);
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
    if (!input_tensors_[arg.index]->CopyDataToTensor(buffer, length)) {
      NNADAPTER_LOG(FATAL) << "Failed to copy data for input " << arg.index
                           << "th !";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    }
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK(graph_->Run());
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from tim-vx
    // according to the dynamic dimensions of the inputs, fill them to 'type'
    // and call the 'access' function to re-allocate the host output memory
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    if (!output_tensors_[arg.index]->CopyDataFromTensor(buffer)) {
      NNADAPTER_LOG(FATAL) << "Failed to copy data for output " << arg.index
                           << "th !";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    }
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
