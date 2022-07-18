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

#include "driver/amlogic_npu/engine.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "driver/amlogic_npu/converter/converter.h"
#include "driver/amlogic_npu/optimizer/unpack_op_fusion.h"
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
namespace amlogic_npu {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from amlnpu_ddk
}

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
  if (model && cache->dir && cache->token) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->token);
  }
  dump_graph_buffer_ = &cache->buffer;
  return cache->buffer.empty() ? BuildFromModel(model) : BuildFromCache(cache);
}

int Program::BuildFromModel(core::Model* model) {
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseConv2DBatchNormIntoConv2D(model);
  FuseConv2DAddIntoConv2D(model);
  FuseConv2DActivationIntoConv2D(model);
  FuseMatMulAddIntoFullyConnected(model);
  FuseReshapeTransposeReshapeIntoChannelShuffle(model);
  UnpackOpFusion(model);
  ConvertQuantizationSymmToAsymm(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a amlnpu graph
  graph_from_cache_ = false;
  if (!dump_graph_path_.empty()) {
    graph_ = std::make_shared<aml::nn::Graph>(
        const_cast<char*>(dump_graph_path_.c_str()));
    if (!graph_) {
      return NNADAPTER_OUT_OF_MEMORY;
    }
  } else {
    graph_ = std::make_shared<aml::nn::Graph>();
    if (!graph_) {
      return NNADAPTER_OUT_OF_MEMORY;
    }
  }
  Converter converter(graph_.get(), &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors;
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
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors(output_count);
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
  execution_ = std::make_shared<aml::nn::Exection>(graph_.get());
  execution_->Build();
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  graph_ = std::make_shared<aml::nn::Graph>();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  // Load graph from cache buffer
  if (graph_from_cache_ == false) {
    if (graph_->LoadCache(reinterpret_cast<char*>(cache->buffer.data()),
                          cache->buffer.size()) != aml::nn::AML_SUCCESS) {
      NNADAPTER_LOG(FATAL) << "Failed to load cache graph from buffer!";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    } else {
      NNADAPTER_VLOG(3) << "first time to Load cache !";
      graph_from_cache_ = true;
    }
  }
  NNADAPTER_VLOG(3) << "Load cache graph from buffer success.";
  // Indentify the inputs and outputs
  auto input_count = cache->input_types.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_types_ = cache->input_types;
    for (size_t i = 0; i < input_count; i++) {
      const auto& type = cache->input_types[i];
      input_tensors[i] = CreateAmlTensor(
          graph_.get(), string_format("model_input_%d", i), &type);
      NNADAPTER_CHECK(input_tensors[i]);
    }
  }
  auto output_count = cache->output_types.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors(output_count);
  output_types_ = cache->output_types;
  for (size_t i = 0; i < output_count; i++) {
    const auto& type = cache->output_types[i];
    output_tensors[i] = CreateAmlTensor(
        graph_.get(), string_format("model_output_%d", i), &type);
    NNADAPTER_CHECK(output_tensors[i]);
  }
  // graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-specific program.
  execution_ = std::make_shared<aml::nn::Exection>(graph_.get());
  // execution_->Build();
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
  std::vector<aml::nn::InputInfo> input_info(input_count);
  std::vector<aml::nn::OutputInfo> output_info(output_count);
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
    input_info[arg.index].pass_through = false;
    input_info[arg.index].type =
        static_cast<int>(ConvertToAmlPrecisionType(type.precision));
    input_info[arg.index].layout =
        static_cast<int>(ConvertToAmlDataLayoutType(type.layout));
    if (graph_from_cache_ == true) {
      execution_->SwapIObuffer(0, i, reinterpret_cast<char*>(buffer));
      NNADAPTER_LOG(INFO) << "Input SwapIObuffer " << buffer;
    }
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from amlnpu_ddk
    // according to the dynamic dimensions of the inputs, fill them to 'type'
    // and call the 'access' function to re-allocate the host output memory
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    // Initialize the output info for the execution
    output_info[arg.index].index = arg.index;
    output_info[arg.index].buf = buffer;
    output_info[arg.index].size = length;
    output_info[arg.index].want_float = false;
    output_info[arg.index].type =
        static_cast<int>(ConvertToAmlPrecisionType(type->precision));
    output_info[arg.index].layout =
        static_cast<int>(ConvertToAmlDataLayoutType(type->layout));
    if (graph_from_cache_ == true) {
      execution_->SwapIObuffer(1, i, reinterpret_cast<char*>(buffer));
      NNADAPTER_LOG(INFO) << "Output SwapIObuffer " << buffer;
    }
  }
  auto start_time = GetCurrentUS();

  NNADAPTER_CHECK_EQ(execution_->SetInputs(input_info), aml::nn::AML_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->Run(), aml::nn::AML_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->GetOutputs(output_info), aml::nn::AML_SUCCESS);

  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto buffer = output_info[i].buf;
    auto length = output_info[i].size;
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
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

}  // namespace amlogic_npu
}  // namespace nnadapter
