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
#include <unistd.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "driver/nvidia_tensorrt/optimizer/remove_reshape_before_fully_connected.h"
#include "driver/nvidia_tensorrt/optimizer/unpack_op_fusion.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

Context::Context(void* device, const char* properties) : device_(device) {}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  device_buffers_.clear();
  tensors_.clear();
  input_indices_.clear();
  output_indices_.clear();
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
  return NNADAPTER_NO_ERROR;
}

// Only remain min/opt/max shapes
static void ConvertDynamicDimensions(NNAdapterOperandType* type) {
  if (type->dimensions.dynamic_count == 0) return;
  int count = type->dimensions.count;
  int dynamic_count = type->dimensions.dynamic_count;
  auto& dynamic_data = type->dimensions.dynamic_data;
  std::vector<int32_t> opt_shape(dynamic_data[0], dynamic_data[0] + count);
  std::vector<int32_t> min_shape(opt_shape);
  std::vector<int32_t> max_shape(opt_shape);
  for (int i = 1; i < dynamic_count; i++) {
    for (int j = 0; j < count; j++) {
      if (dynamic_data[i][j] < min_shape[j]) {
        min_shape[j] = dynamic_data[i][j];
      }
      if (dynamic_data[i][j] > max_shape[j]) {
        max_shape[j] = dynamic_data[i][j];
      }
    }
  }
  memcpy(dynamic_data[0], min_shape.data(), sizeof(int32_t) * count);
  memcpy(dynamic_data[1], opt_shape.data(), sizeof(int32_t) * count);
  memcpy(dynamic_data[2], max_shape.data(), sizeof(int32_t) * count);
  type->dimensions.dynamic_count = 3;
}

void Program::CompleteConfig(core::Model* model) {
  config_.reset(builder_->createBuilderConfig());
  NNADAPTER_CHECK(config_);
  if (with_dynamic_shape_) {
    for (auto operand : model->input_operands) {
      auto type = operand->type;
      auto& dimensions = type.dimensions;
      if (dimensions.dynamic_count == 0) continue;
      ConvertDynamicDimensions(&type);
      NNADAPTER_CHECK_EQ(dimensions.dynamic_count, 3);
      // need not to delete by user
      auto profile = builder_->createOptimizationProfile();
      auto name = tensors_.at(operand).back()->getName();
      nvinfer1::Dims dims;
      dims.nbDims = dimensions.count;
      memcpy(dims.d, dimensions.dynamic_data[0], sizeof(int32_t) * dims.nbDims);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dims);
      memcpy(dims.d, dimensions.dynamic_data[1], sizeof(int32_t) * dims.nbDims);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims);
      memcpy(dims.d, dimensions.dynamic_data[2], sizeof(int32_t) * dims.nbDims);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dims);
      config_->addOptimizationProfile(profile);
    }
  }
}

// Malloc gpu memory according to max dims
static void MallocMaxDeviceMemory(
    NNAdapterOperandType* type,
    std::vector<std::shared_ptr<void>>* device_buffers,
    int idx) {
  // Get max tensor size
  int64_t size = 1;
  auto& dims = type->dimensions;
  if (dims.dynamic_count == 0) {
    size = ProductionOfDimensions(dims.data, dims.count);
  } else {
    NNADAPTER_CHECK_EQ(dims.dynamic_count, 3U);
    size = ProductionOfDimensions(dims.dynamic_data[2], dims.count);
  }
  NNADAPTER_VLOG(5) << "Malloc max gpu memory size: " << size;
  // Malloc gpu memory
  void* data_ptr{nullptr};
  size *= GetOperandPrecisionDataLength(type->precision);
  NNADAPTER_CHECK_EQ(cudaMalloc(&data_ptr, size), cudaSuccess);
  std::shared_ptr<void> device_buffer(data_ptr, [](void* ptr) {
    NNADAPTER_CHECK_EQ(cudaFree(ptr), cudaSuccess);
  });
  device_buffers->at(idx) = std::move(device_buffer);
}

int Program::BuildFromModel(core::Model* model) {
  for (auto operand : model->input_operands) {
    if (IsOperandWithDynamicShape(operand)) {
      with_dynamic_shape_ = true;
      break;
    }
  }
  // Optimize the model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  UnpackOpFusion(model);
  FuseMatMulAddIntoFullyConnected(model);
  RemoveReshapeBeforeFullyConnected(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Create builder_, network_
  nvinfer1::ILogger& logger = *TrtLogger::Global();
  builder_.reset(nvinfer1::createInferBuilder(logger));
  NNADAPTER_CHECK(builder_);
  network_.reset(builder_->createNetworkV2(1U));
  NNADAPTER_CHECK(network_);
  // Convert a NNAdapter model to a nv network
  Converter converter(network_.get(), &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Mark output
  for (auto operand : model->output_operands) {
    NNADAPTER_CHECK(tensors_.count(operand));
    auto tensor = tensors_.at(operand).back();
    tensor->setName(OperandToString(operand).c_str());
    network_->markOutput(*tensor);
  }
  // Create config_ and set options
  CompleteConfig(model);
// Create execute context
#if TENSORRT_MAJOR_VERSION >= 8
  plan_.reset(builder_->buildSerializedNetwork(*network_, *config_));
  NNADAPTER_CHECK(plan_);
  runtime_.reset(nvinfer1::createInferRuntime(*TrtLogger::Global()));
  NNADAPTER_CHECK(runtime_);
  engine_.reset(runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
#else
  engine_.reset(builder_->buildEngineWithConfig(*network_, *config_));
#endif
  NNADAPTER_CHECK(engine_);
  execution_context_.reset(engine_->createExecutionContext());
  NNADAPTER_CHECK(execution_context_);
  // Identify the inputs and outputs
  size_t input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  input_types_.resize(input_count);
  size_t output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  output_types_.resize(output_count);
  NNADAPTER_CHECK_EQ(engine_->getNbBindings(), input_count + output_count);
  device_buffers_.resize(input_count + output_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    input_types_[i] = operand->type;
    ConvertDynamicDimensions(&input_types_[i]);
    input_indices_.push_back(
        engine_->getBindingIndex(tensors_.at(operand).back()->getName()));
    MallocMaxDeviceMemory(
        &input_types_[i], &device_buffers_, input_indices_[i]);
  }
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    output_types_[i] = operand->type;
    ConvertDynamicDimensions(&output_types_[i]);
    output_indices_.push_back(
        engine_->getBindingIndex(tensors_.at(operand).back()->getName()));
    MallocMaxDeviceMemory(
        &output_types_[i], &device_buffers_, output_indices_[i]);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  NNADAPTER_LOG(FATAL) << "Build from cache is unimpleted.";
  return NNADAPTER_DEVICE_INTERNAL_ERROR;
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
    arg.access(arg.memory, &type);
    // Check dimensions count
    uint32_t count = type.dimensions.count;
    auto& src_dimensions = input_types_[i].dimensions;
    if (count != src_dimensions.count) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
    // Check dimensions data
    bool is_matched = true;
    int32_t* data = type.dimensions.data;
    int32_t* src_data = src_dimensions.data;
    for (uint32_t j = 0; j < count; j++) {
      if (data[j] != src_data[j]) {
        is_matched = false;
        break;
      }
    }
    if (is_matched) continue;
    // Check dynamic dymensions data
    NNADAPTER_CHECK_EQ(src_dimensions.dynamic_count, 3U);
    for (uint32_t j = 0; j < count; j++) {
      if (data[j] < src_dimensions.dynamic_data[1][j] ||
          data[j] > src_dimensions.dynamic_data[2][j]) {
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
  std::vector<void*> device_ptrs;
  for (auto& device_buffer : device_buffers_) {
    device_ptrs.push_back(device_buffer.get());
  }
  // Feed inputs
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    auto type = input_types_[arg.index];
    auto host_ptr = arg.access(arg.memory, &type);
    auto length = GetOperandTypeBufferLength(type);
    NNADAPTER_CHECK_EQ(cudaMemcpy(device_ptrs.at(input_indices_.at(i)),
                                  host_ptr,
                                  length,
                                  cudaMemcpyHostToDevice),
                       cudaSuccess);
    nvinfer1::Dims dims;
    dims.nbDims = type.dimensions.count;
    memcpy(dims.d, type.dimensions.data, dims.nbDims * sizeof(int32_t));
    execution_context_->setBindingDimensions(input_indices_[i], dims);
  }
  NNADAPTER_CHECK(execution_context_->allInputDimensionsSpecified());
  // Execute model
  execution_context_->execute(1, device_ptrs.data());
  // Fetch outputs
  for (uint32_t i = 0; i < output_count; i++) {
    NNAdapterOperandType type = output_types_[i];
    auto dims = execution_context_->getBindingDimensions(output_indices_.at(i));
    type.dimensions.count = dims.nbDims;
    memcpy(type.dimensions.data, dims.d, dims.nbDims * sizeof(int32_t));
    auto& arg = output_arguments[i];
    auto host_ptr = arg.access(arg.memory, &type);
    auto length = GetOperandTypeBufferLength(type);
    NNADAPTER_CHECK_EQ(cudaMemcpy(host_ptr,
                                  device_ptrs.at(output_indices_.at(i)),
                                  length,
                                  cudaMemcpyDeviceToHost),
                       cudaSuccess);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
