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

#include "driver/eeasytech_npu/engine.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "driver/eeasytech_npu/converter/converter.h"
#include "driver/eeasytech_npu/optimizer/unpack_op_fusion.h"
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
namespace eeasytech_npu {

Context::Context(void* device, const char* properties) : device_(device) {}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
  dump_graph_path_ = "";
  dump_graph_buffer_ = nullptr;
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  NNADAPTER_LOG(INFO) << "eeasytech Build.";
  if (model && cache->dir && cache->token) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->token);
  }
  dump_graph_buffer_ = &cache->buffer;
  return cache->buffer.empty() ? BuildFromModel(model) : BuildFromCache(cache);
}

int Program::BuildFromModel(core::Model* model) {
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  UnpackOpFusion(model);
  FuseConv2DBatchNormIntoConv2D(model);
  FuseConv2DAddIntoConv2D(model);
  FuseConv2DActivationIntoConv2D(model);
  FuseMatMulAddIntoFullyConnected(model);
  FuseReshapeTransposeReshapeIntoChannelShuffle(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);

  // Convert a NNAdapter model to a eeasynpu graph
  graph_ = create_graph();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  if (!dump_graph_path_.empty()) {
    if (graph_->EnableCreateCache(dump_graph_path_) == eeasy::nn::EZ_SUCCESS) {
      NNADAPTER_VLOG(3) << "Dump the graph to " << dump_graph_path_
                        << " when the first run";
    } else {
      NNADAPTER_LOG(WARNING) << "Failed to dump the graph to "
                             << dump_graph_path_;
    }
  }
  Converter converter(graph_.get(), &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors;
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
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    NNADAPTER_CHECK(output_tensors[i]);
    output_types_[i] = type;
  }
  graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-related program.
  execution_ = create_exection(graph_);
  execution_->Build();
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  graph_ = create_graph();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  NNADAPTER_LOG(INFO) << "BuildFromCache.";
  // Load graph from cache buffer
  if (graph_->LoadCache(reinterpret_cast<const char*>(cache->buffer.data()),
                        cache->buffer.size()) != eeasy::nn::EZ_SUCCESS) {
    NNADAPTER_LOG(FATAL) << "Failed to load cache graph from buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  NNADAPTER_VLOG(3) << "Load cache graph from buffer success.";
  // Indentify the inputs and outputs
  auto input_count = cache->input_types.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_types_ = cache->input_types;
    for (size_t i = 0; i < input_count; i++) {
      const auto& type = cache->input_types[i];
      input_tensors[i] = CreateEznnTensor(
          graph_.get(), string_format("model_input_%d", i), &type);
      NNADAPTER_CHECK(input_tensors[i]);
    }
  }
  auto output_count = cache->output_types.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors(output_count);
  output_types_ = cache->output_types;
  for (size_t i = 0; i < output_count; i++) {
    const auto& type = cache->output_types[i];
    output_tensors[i] = CreateEznnTensor(
        graph_.get(), string_format("model_output_%d", i), &type);
    NNADAPTER_CHECK(output_tensors[i]);
  }
  graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-specific program.
  execution_ = create_exection(graph_);
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
  std::vector<eeasy::nn::InputInfo> input_info(input_count);
  std::vector<eeasy::nn::OutputInfo> output_info(output_count);
  NNADAPTER_LOG(INFO) << "input_info.size " << input_info.size();
  NNADAPTER_LOG(INFO) << "output_info.size " << output_info.size();
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
    // Initialize the input info for the execution
    input_info[arg.index].index = arg.index;
    input_info[arg.index].buf = buffer;
    input_info[arg.index].size = length;
    for (int d = 0; d < input_types_[arg.index].dimensions.count; d++) {
      input_info[arg.index].dims.push_back(
          input_types_[arg.index].dimensions.data[d]);
    }
    input_info[arg.index].pass_through = false;
    input_info[arg.index].type = ConvertToEznnPrecisionType(type.precision);
    input_info[arg.index].layout = ConvertToEznnDataLayoutType(type.layout);
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    // Initialize the output info for the execution
    output_info[arg.index].index = arg.index;
    output_info[arg.index].buf = buffer;
    output_info[arg.index].size = length;
    output_info[arg.index].want_float = false;
    output_info[arg.index].type = ConvertToEznnPrecisionType(type->precision);
    output_info[arg.index].layout = ConvertToEznnDataLayoutType(type->layout);
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK_EQ(execution_->SetInputs(input_info), eeasy::nn::EZ_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->Run(), eeasy::nn::EZ_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->GetOutputs(output_info),
                     eeasy::nn::EZ_SUCCESS);
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  // Read data from the dump graph file and fill to cache
  if (!dump_graph_path_.empty()) {
    if (ReadFile(dump_graph_path_, dump_graph_buffer_)) {
      NNADAPTER_LOG(INFO) << "Read the dump graph file " << dump_graph_path_
                          << " success.";
    } else {
      NNADAPTER_LOG(INFO) << "Failed to read the dump graph file "
                          << dump_graph_path_ << "!";
    }
    dump_graph_path_ = "";
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
