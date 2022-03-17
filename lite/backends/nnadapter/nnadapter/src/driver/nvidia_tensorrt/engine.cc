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
  size *= GetOperandPrecisionDataLength(type->precision);
  NNADAPTER_VLOG(5) << "Malloc max gpu memory size: " << size;
  // Malloc gpu memory
  void* data_ptr{nullptr};
  NNADAPTER_CHECK_EQ(cudaMalloc(&data_ptr, size), cudaSuccess);
  std::shared_ptr<void> device_buffer(data_ptr, [](void* ptr) {
    NNADAPTER_CHECK_EQ(cudaFree(ptr), cudaSuccess);
  });
  device_buffers->at(idx) = std::move(device_buffer);
}

static core::Argument* FindArgumentByIndex(core::Argument* arguments,
                                           int index,
                                           uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    if (arguments[i].index == index) {
      return &arguments[i];
    }
  }
  return static_cast<core::Argument*>(nullptr);
}

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_VLOG(1) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  // Device type
  std::string device_type = key_values.count(NVIDIA_TENSORRT_DEVICE_TYPE)
                                ? key_values[NVIDIA_TENSORRT_DEVICE_TYPE]
                                : GetStringFromEnv(NVIDIA_TENSORRT_DEVICE_TYPE);
  if (!device_type.empty()) {
    if (device_type == "GPU") {
      device_type_ = nvinfer1::DeviceType::kGPU;
    } else if (device_type == "DLA") {
      device_type_ = nvinfer1::DeviceType::kDLA;
    } else {
      NNADAPTER_LOG(FATAL) << "Not support NVIDIA_TENSORRT_DEVICE_TYPE: "
                           << device_type;
    }
  }
  // Device id
  device_id_ = key_values.count(NVIDIA_TENSORRT_DEVICE_ID)
                   ? atoi(key_values[NVIDIA_TENSORRT_DEVICE_ID].c_str())
                   : GetIntFromEnv(NVIDIA_TENSORRT_DEVICE_ID);
  device_id_ = device_id_ > 0 ? device_id_ : 0;
  // Precision
  std::string precision = key_values.count(NVIDIA_TENSORRT_PRECISION)
                              ? key_values[NVIDIA_TENSORRT_PRECISION]
                              : GetStringFromEnv(NVIDIA_TENSORRT_PRECISION);
  if (!precision.empty()) {
    if (precision == "float32") {
      precision_ = kFloat32;
    } else if (precision == "float16") {
      precision_ = kFloat16;
    } else if (precision == "int8") {
      precision_ = kInt8;
    } else {
      NNADAPTER_LOG(FATAL) << "Not support NVIDIA_TENSORRT_PRECISION: "
                           << precision;
    }
  }
  // Fallback
  gpu_fallback_ = key_values.count(NVIDIA_TENSORRT_GPU_FALLBACK)
                      ? key_values[NVIDIA_TENSORRT_GPU_FALLBACK] == "1"
                      : GetBoolFromEnv(NVIDIA_TENSORRT_GPU_FALLBACK, true);
  // Calibration dataset
  calibration_dataset_path_ =
      key_values.count(NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH)
          ? key_values[NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH]
          : GetStringFromEnv(NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH);
  // Calibration table
  calibration_table_path_ =
      key_values.count(NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH)
          ? key_values[NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH]
          : GetStringFromEnv(NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH);
  if (precision_ == kInt8) {
    NNADAPTER_CHECK(!calibration_dataset_path_.empty() ||
                    !calibration_table_path_.empty())
        << "Either NVIDIA_TENSORRT_CALIBRATION_DATASET_PATH or "
           "NVIDIA_TENSORRT_CALIBRATION_TABLE_PATH should be set if precision "
           "is int8.";
  }
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  device_buffers_.clear();
  tensors_.clear();
  input_indices_.clear();
  output_indices_.clear();
  input_types_.clear();
  output_types_.clear();
  weight_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  // 1. Build model to engine_
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(BuildFromModel(model), NNADAPTER_NO_ERROR);
    cache->buffer.resize(plan_->size());
    memcpy(cache->buffer.data(), plan_->data(), sizeof(int8_t) * plan_->size());
  } else {
    NNADAPTER_CHECK_EQ(BuildFromCache(cache), NNADAPTER_NO_ERROR);
  }
  // 2. Create execution_context_
  execution_context_.reset(engine_->createExecutionContext());
  NNADAPTER_CHECK(execution_context_);
  // 3. Prepare device_buffers_
  size_t input_count = input_types_.size();
  size_t output_count = output_types_.size();
  NNADAPTER_CHECK_EQ(static_cast<size_t>(engine_->getNbBindings()),
                     input_count + output_count);
  device_buffers_.resize(input_count + output_count);
  for (size_t i = 0; i < input_count; i++) {
    std::string name = "input" + std::to_string(i);
    input_indices_.push_back(engine_->getBindingIndex(name.c_str()));
    MallocMaxDeviceMemory(
        &input_types_.at(i), &device_buffers_, input_indices_.at(i));
  }
  for (size_t i = 0; i < output_count; i++) {
    std::string name = "output" + std::to_string(i);
    output_indices_.push_back(engine_->getBindingIndex(name.c_str()));
    MallocMaxDeviceMemory(
        &output_types_.at(i), &device_buffers_, output_indices_.at(i));
  }
  return NNADAPTER_NO_ERROR;
}

void Program::CompleteConfig(core::Model* model) {
  config_.reset(builder_->createBuilderConfig());
  NNADAPTER_CHECK(config_);
  // Set dynamic shapes
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
  // Set device_type
  auto device_type = context_->DeviceType();
  config_->setDefaultDeviceType(device_type);
  // Set device_id
  if (device_type == nvinfer1::DeviceType::kDLA) {
    int device_id = context_->DeviceId();
    if (builder_->getNbDLACores() > device_id) {
      config_->setDLACore(device_id);
      NNADAPTER_VLOG(1) << "Tring to use DLA core " << device_id;
    } else {
      NNADAPTER_LOG(WARNING) << "Trying to use DLA core " << device_id
                             << " failed. The platform only has "
                             << builder_->getNbDLACores() << " DLA cores.";
    }
  }
  // Set precision
  PrecisionMode precision = context_->Precision();
  switch (precision) {
    case kFloat32:
      if (device_type == nvinfer1::DeviceType::kDLA) {
        NNADAPTER_LOG(WARNING) << "Only support float16 or int8 if device type "
                                  "is DLA. Float16 is selected by default.";
        config_->setFlag(nvinfer1::BuilderFlag::kFP16);
      }
      break;
    case kFloat16:
      config_->setFlag(nvinfer1::BuilderFlag::kFP16);
      break;
    case kInt8:
      config_->setFlag(nvinfer1::BuilderFlag::kINT8);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Not support precision mode: "
                           << static_cast<int>(precision);
      break;
  }
  // Set fallback
  if (context_->GpuFallback()) {
    config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  }
  // Calibration
  if (precision == kInt8) {
    NNADAPTER_CHECK(!with_dynamic_shape_)
        << "Int8 and dynamic shape is incompatible.";
    calibrator_.reset(new Int8EntropyCalibrator(
        model->input_operands.at(0)->type.dimensions.data[0],
        context_->CalibrationDatasetPath(),
        context_->CalibrationTablePath()));
    config_->setInt8Calibrator(calibrator_.get());
  }
}

int Program::BuildFromModel(core::Model* model) {
  for (auto operand : model->input_operands) {
    if (IsOperandWithDynamicShape(operand)) {
      with_dynamic_shape_ = true;
      break;
    }
  }
  // 1. Optimize the model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  UnpackOpFusion(model);
  FuseMatMulAddIntoFullyConnected(model);
  RemoveReshapeBeforeFullyConnected(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // 2. Build model, serialize to plan_, create engnie_
  builder_.reset(nvinfer1::createInferBuilder(*TrtLogger::Global()));
  NNADAPTER_CHECK(builder_);
  if (context_->Precision() == kInt8) {
    network_.reset(builder_->createNetworkV2(0U));
  } else {
    network_.reset(builder_->createNetworkV2(
        1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  }
  NNADAPTER_CHECK(network_);
  // Convert a NNAdapter model to a tensorrt network
  Converter converter(network_.get(), &tensors_, &weight_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Create config_ and set options
  CompleteConfig(model);
// Serialize to plan_
#if TENSORRT_MAJOR_VERSION >= 8
  plan_.reset(builder_->buildSerializedNetwork(*network_, *config_));
  NNADAPTER_CHECK(plan_);
  runtime_.reset(nvinfer1::createInferRuntime(*TrtLogger::Global()));
  NNADAPTER_CHECK(runtime_);
  engine_.reset(runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
  NNADAPTER_CHECK(engine_);
#else
  engine_.reset(builder_->buildEngineWithConfig(*network_, *config_));
  NNADAPTER_CHECK(engine_);
  plan_.reset(engine_->serialize());
  NNADAPTER_CHECK(plan_);
#endif
  // 3. Identify the inputs and outputs
  size_t input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  input_types_.resize(input_count);
  size_t output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  output_types_.resize(output_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands.at(i);
    input_types_.at(i) = operand->type;
    ConvertDynamicDimensions(&input_types_.at(i));
  }
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands.at(i);
    output_types_.at(i) = operand->type;
    ConvertDynamicDimensions(&output_types_.at(i));
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  // 1. Create engine_
  runtime_.reset(nvinfer1::createInferRuntime(*TrtLogger::Global()));
  NNADAPTER_CHECK(runtime_);
  engine_.reset(runtime_->deserializeCudaEngine(
      reinterpret_cast<void*>(cache->buffer.data()), cache->buffer.size()));
  NNADAPTER_CHECK(engine_);
  // 2. Identify the inputs and outputs
  input_types_ = cache->input_types;
  for (size_t i = 0; i < input_types_.size(); i++) {
    ConvertDynamicDimensions(&input_types_.at(i));
  }
  output_types_ = cache->output_types;
  for (size_t i = 0; i < output_types_.size(); i++) {
    ConvertDynamicDimensions(&output_types_.at(i));
  }
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get actual type
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNAdapterOperandType type;
    arg->access(arg->memory, &type);
    // Check dimensions count
    uint32_t count = type.dimensions.count;
    auto& src_dimensions = input_types_.at(i).dimensions;
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
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    auto type = input_types_.at(i);
    auto host_ptr = arg->access(arg->memory, &type);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(type);
    NNADAPTER_CHECK_EQ(cudaMemcpy(device_ptrs.at(input_indices_.at(i)),
                                  host_ptr,
                                  length,
                                  cudaMemcpyHostToDevice),
                       cudaSuccess);
    nvinfer1::Dims dims;
    dims.nbDims = type.dimensions.count;
    memcpy(dims.d, type.dimensions.data, dims.nbDims * sizeof(int32_t));
    execution_context_->setBindingDimensions(input_indices_.at(i), dims);
  }
  NNADAPTER_CHECK(execution_context_->allInputDimensionsSpecified());
  // Execute model
  execution_context_->execute(1, device_ptrs.data());
  // Fetch outputs
  for (uint32_t i = 0; i < output_count; i++) {
    auto arg = FindArgumentByIndex(output_arguments, i, output_count);
    NNADAPTER_CHECK(arg) << "Output argument " << i << " does not exist!";
    NNAdapterOperandType type = output_types_.at(i);
    auto dims = execution_context_->getBindingDimensions(output_indices_.at(i));
    type.dimensions.count = dims.nbDims;
    memcpy(type.dimensions.data, dims.d, dims.nbDims * sizeof(int32_t));
    auto host_ptr = arg->access(arg->memory, &type);
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
