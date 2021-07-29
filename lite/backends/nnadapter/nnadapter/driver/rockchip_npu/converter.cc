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

#include "driver/rockchip_npu/converter.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "driver/rockchip_npu/optimizer/fix_ops.h"
#include "driver/rockchip_npu/optimizer/unpack_op_fusion.h"
#include "optimizer/symm2asymm.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace rockchip_npu {

Context::Context(void* device, const char* properties) : device_(device) {
  // TODO(hong19860320) create the raw context from rknpu_ddk
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  input_info_.clear();
  output_info_.clear();
  input_zero_points_.clear();
  output_zero_points_.clear();
  dump_graph_path_ = "";
  dump_graph_buffer_ = nullptr;
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();
  if (model && cache->dir && cache->key) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->key);
  }
  dump_graph_buffer_ = &cache->buffer;
  return cache->buffer.empty() ? BuildFromModel(model) : BuildFromCache(cache);
}

int Program::BuildFromModel(hal::Model* model) {
  // Convert the quantization parameters of the operands in the NNAdapter model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  UnpackOpFusion(model);
  FixOps(model);
  ConvertQuantizationSymmToAsymm(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Convert a NNAdapter model to a rknpu graph
  graph_ = std::make_shared<rk::nn::Graph>();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  if (!dump_graph_path_.empty()) {
    if (graph_->EnableCreateCache(dump_graph_path_) == rk::nn::RK_SUCCESS) {
      NNADAPTER_VLOG(3) << "Dump the graph to " << dump_graph_path_
                        << " when the first run";
    } else {
      NNADAPTER_LOG(WARNING) << "Failed to dump the graph to "
                             << dump_graph_path_;
    }
  }
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_CONCAT:
        ConvertConcat(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertPool2D(operation);
        break;
      case NNADAPTER_ADD:
      case NNADAPTER_SUB:
      case NNADAPTER_MUL:
      case NNADAPTER_DIV:
        ConvertElementwise(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
        break;
      case NNADAPTER_SIGMOID:
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
        ConvertActivation(operation);
        break;
      case NNADAPTER_RESHAPE:
        ConvertReshape(operation);
        break;
      case NNADAPTER_TRANSPOSE:
        ConvertTranspose(operation);
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
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_info_.resize(input_count);
    input_zero_points_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
      input_tensors[i] = tensors_[operand].back();
      NNADAPTER_CHECK(input_tensors[i]);
      // Initialize the input info for the execution
      input_info_[i].index = i;
      input_info_[i].buf = nullptr;
      input_info_[i].size = 0;
      input_info_[i].pass_through = false;
      input_info_[i].type = ConvertPrecision(type.precision);
      input_info_[i].layout = ConvertDataLayout(type.layout);
      input_zero_points_[i] = IsUInt8AsymmPerLayerQuantization(type.precision)
                                  ? type.asymm_per_layer_params.zero_point
                                  : 0;
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_tensors;
  output_tensors.resize(output_count);
  output_info_.resize(output_count);
  output_zero_points_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    NNADAPTER_CHECK(output_tensors[i]);
    // Initialize the output info for the execution
    output_info_[i].index = i;
    output_info_[i].buf = nullptr;
    output_info_[i].size = 0;
    output_info_[i].want_float = false;
    output_info_[i].type = ConvertPrecision(type.precision);
    output_info_[i].layout = ConvertDataLayout(type.layout);
    output_zero_points_[i] = IsUInt8AsymmPerLayerQuantization(type.precision)
                                 ? type.asymm_per_layer_params.zero_point
                                 : 0;
  }
  graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-related program.
  execution_ = std::make_shared<rk::nn::Exection>(graph_.get());
  execution_->Build();
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(hal::Cache* cache) {
  graph_ = std::make_shared<rk::nn::Graph>();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  // Load graph from cache buffer
  if (graph_->LoadCache(reinterpret_cast<const char*>(cache->buffer.data()),
                        cache->buffer.size()) != rk::nn::RK_SUCCESS) {
    NNADAPTER_LOG(FATAL) << "Failed to load cache graph from buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  NNADAPTER_VLOG(3) << "Load cache graph from buffer success.";
  // Indentify the inputs and outputs
  auto input_count = cache->input_types.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_info_.resize(input_count);
    input_zero_points_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      // Add a input tensor
      const auto& type = cache->input_types[i];
      input_tensors[i] = AddTensor(string_format("model_input_%d", i), &type);
      NNADAPTER_CHECK(input_tensors[i]);
      // Initialize the input info for the execution
      input_info_[i].index = i;
      input_info_[i].buf = nullptr;
      input_info_[i].size = 0;
      input_info_[i].pass_through = false;
      input_info_[i].type = ConvertPrecision(type.precision);
      input_info_[i].layout = ConvertDataLayout(type.layout);
      input_zero_points_[i] = IsUInt8AsymmPerLayerQuantization(type.precision)
                                  ? type.asymm_per_layer_params.zero_point
                                  : 0;
    }
  }
  auto output_count = cache->output_types.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_tensors;
  output_tensors.resize(output_count);
  output_info_.resize(output_count);
  output_zero_points_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    // Add a output tensor
    const auto& type = cache->output_types[i];
    output_tensors[i] = AddTensor(string_format("model_output_%d", i), &type);
    NNADAPTER_CHECK(output_tensors[i]);
    // Initialize the output info for the execution
    output_info_[i].index = i;
    output_info_[i].buf = nullptr;
    output_info_[i].size = 0;
    output_info_[i].want_float = false;
    output_info_[i].type = ConvertPrecision(type.precision);
    output_info_[i].layout = ConvertDataLayout(type.layout);
    output_zero_points_[i] = IsUInt8AsymmPerLayerQuantization(type.precision)
                                 ? type.asymm_per_layer_params.zero_point
                                 : 0;
  }
  graph_->SetInputsOutputs(input_tensors, output_tensors);
  // Create an execution to build the graph to the device-related program.
  execution_ = std::make_shared<rk::nn::Exection>(graph_.get());
  execution_->Build();
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  NNADAPTER_CHECK_EQ(input_info_.size(), input_count);
  NNADAPTER_CHECK_EQ(output_info_.size(), output_count);
  for (uint32_t i = 0; i < input_count; i++) {
    auto& argument = input_arguments[i];
    auto buffer = reinterpret_cast<uint8_t*>(argument.buffer);
    auto zero_point = input_zero_points_[argument.index];
    Symm2AsymmData(reinterpret_cast<const int8_t*>(argument.buffer),
                   argument.length,
                   zero_point,
                   buffer);
    input_info_[argument.index].buf = argument.buffer;
    input_info_[argument.index].size = argument.length;
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
    output_info_[argument.index].buf = argument.buffer;
    output_info_[argument.index].size = argument.length;
  }
  NNADAPTER_CHECK_EQ(execution_->SetInputs(input_info_), rk::nn::RK_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->Run(), rk::nn::RK_SUCCESS);
  NNADAPTER_CHECK_EQ(execution_->GetOutputs(output_info_), rk::nn::RK_SUCCESS);
  for (uint32_t i = 0; i < output_count; i++) {
    auto& argument = output_arguments[i];
    auto buffer = reinterpret_cast<int8_t*>(argument.buffer);
    auto zero_point = output_zero_points_[argument.index];
    Asymm2SymmData(reinterpret_cast<const uint8_t*>(argument.buffer),
                   argument.length,
                   zero_point,
                   buffer);
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

std::string Program::GetTensorName(hal::Operand* operand) {
  auto operand_id = OperandIdToString(operand);
  auto index = 0;
  auto it = tensors_.find(operand);
  if (it != tensors_.end()) {
    index = it->second.size();
  }
  return operand_id + string_format("_%d", index);
}

std::shared_ptr<rk::nn::Tensor> Program::GetMappedTensor(
    hal::Operand* operand) {
  auto it = tensors_.find(operand);
  if (it != tensors_.end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<rk::nn::Tensor> Program::UpdateTensorMap(
    hal::Operand* operand, std::shared_ptr<rk::nn::Tensor> tensor) {
  auto it = tensors_.find(operand);
  if (it == tensors_.end()) {
    auto result = tensors_.insert(std::make_pair(
        operand, std::vector<std::shared_ptr<rk::nn::Tensor>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

std::shared_ptr<rk::nn::Tensor> Program::AddTensor(
    const std::string& name,
    int32_t* dimensions,
    uint32_t dimension_count,
    rk::nn::PrecisionType precision,
    const float* quant_scale,
    const int32_t* zero_point,
    void* buffer,
    rk::nn::DataLayoutType layout) {
  auto attr = std::make_shared<rk::nn::TensorAttr>();
  attr->name = name;
  attr->role = buffer ? rk::nn::TensorRole::CONST : rk::nn::TensorRole::VAR;
  attr->dims = ConvertDimensions(dimensions, dimension_count);
  attr->precision = precision;
  attr->layout = layout;
  if (quant_scale) {
    // Quantization types
    if (precision == rk::nn::PrecisionType::UINT8) {
      attr->qntBits = 8;
    } else if (precision == rk::nn::PrecisionType::INT32) {
      attr->qntBits = 32;
    } else {
      NNADAPTER_LOG(FATAL)
          << "Only UINT8 and INT32 is supported for quantizaion.";
    }
    if (zero_point) {
      attr->qntType = rk::nn::QuantizationType::AFFINE_ASYMMETRIC;
      attr->qntParamAffineAsymmetric.scale.resize(1);
      attr->qntParamAffineAsymmetric.scale[0] = *quant_scale;
      attr->qntParamAffineAsymmetric.zero_point.resize(1);
      attr->qntParamAffineAsymmetric.zero_point[0] = *zero_point;
    } else {
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale.resize(1);
      attr->qntParamSymmetric.scale[0] = *quant_scale;
    }
  } else {
    // TODO(hong19860320) Supports the normal types, such as float etc.
    NNADAPTER_LOG(FATAL) << "Only quantizaion types are supported.";
  }
  auto tensor = graph_->CreateTensor(attr, buffer);
  NNADAPTER_CHECK(tensor);
  return tensor;
}

std::shared_ptr<rk::nn::Tensor> Program::AddTensor(
    const std::string& name,
    const NNAdapterOperandType* type,
    void* buffer,
    std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimension_count; i++) {
      dimensions.push_back(type->dimensions[i]);
    }
  }
  auto precision = ConvertPrecision(type->precision);
  auto layout = ConvertDataLayout(type->layout);
  const float* quant_scale = nullptr;
  const int32_t* zero_point = nullptr;
  switch (type->precision) {
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      quant_scale = &type->asymm_per_layer_params.scale;
      zero_point = &type->asymm_per_layer_params.zero_point;
      break;
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
      quant_scale = &type->symm_per_layer_params.scale;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can not add a rk::nn::Tensor with precision="
                           << OperandPrecisionCodeToString(type->precision)
                           << " !";
      break;
  }
  return AddTensor(name,
                   dimensions.data(),
                   dimensions.size(),
                   precision,
                   quant_scale,
                   zero_point,
                   buffer,
                   layout);
}

std::shared_ptr<rk::nn::Tensor> Program::AddConstantTensor(
    void* values,
    int32_t* dimensions,
    uint32_t dimension_count,
    rk::nn::PrecisionType precision,
    const float* quant_scale,
    const int32_t* zero_point) {
  auto name = GetTensorName(nullptr);
  auto tensor = AddTensor(name,
                          dimensions,
                          dimension_count,
                          precision,
                          quant_scale,
                          zero_point,
                          values,
                          rk::nn::DataLayoutType::NCHW);
  NNADAPTER_CHECK(tensor);
  UpdateTensorMap(nullptr, tensor);
  return tensor;
}

std::shared_ptr<rk::nn::Tensor> Program::AddVariableTensor(
    const std::string& name,
    int32_t* dimensions,
    uint32_t dimension_count,
    rk::nn::PrecisionType precision,
    const float* quant_scale,
    const int32_t* zero_point) {
  return AddTensor(name,
                   dimensions,
                   dimension_count,
                   precision,
                   quant_scale,
                   zero_point,
                   nullptr,
                   rk::nn::DataLayoutType::NCHW);
}

std::shared_ptr<rk::nn::Tensor> Program::AddQuant8ConstantTensor(
    uint8_t* values,
    int32_t* dimensions,
    uint32_t dimension_count,
    float quant_scale,
    int32_t zero_point) {
  return AddConstantTensor(values,
                           dimensions,
                           dimension_count,
                           rk::nn::PrecisionType::UINT8,
                           &quant_scale,
                           &zero_point);
}

std::shared_ptr<rk::nn::Tensor> Program::AddQuant32ConstantTensor(
    int32_t* values,
    int32_t* dimensions,
    uint32_t dimension_count,
    float quant_scale) {
  return AddConstantTensor(values,
                           dimensions,
                           dimension_count,
                           rk::nn::PrecisionType::INT32,
                           &quant_scale,
                           nullptr);
}

std::shared_ptr<rk::nn::Tensor> Program::AddQuant8VariableTensor(
    const std::string& name,
    int32_t* dimensions,
    uint32_t dimension_count,
    float quant_scale,
    int32_t zero_point) {
  return AddVariableTensor(name,
                           dimensions,
                           dimension_count,
                           rk::nn::PrecisionType::UINT8,
                           &quant_scale,
                           &zero_point);
}

std::shared_ptr<rk::nn::Tensor> Program::ConvertOperand(
    hal::Operand* operand, std::vector<int32_t> dimensions) {
  auto tensor = AddTensor(
      GetTensorName(operand), &operand->type, operand->buffer, dimensions);
  NNADAPTER_CHECK(tensor);
  // Use to find the tensor based on the pointer of operand
  UpdateTensorMap(operand, tensor);
  return tensor;
}

}  // namespace rockchip_npu
}  // namespace nnadapter
