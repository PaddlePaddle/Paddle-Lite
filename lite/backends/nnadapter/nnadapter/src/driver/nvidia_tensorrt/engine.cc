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
  device_data_.clear();
  tensors_.clear();
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

int Program::BuildFromModel(core::Model* model) {
  // Optimize the model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  UnpackOpFusion(model);
  FuseMatMulAddIntoFullyConnected(model);
  RemoveReshapeBeforeFullyConnected(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Create builder_, network_, config_
  nvinfer1::ILogger& logger = *TrtLogger::Global();
  builder_.reset(nvinfer1::createInferBuilder(logger));
  NNADAPTER_CHECK(builder_);
  network_.reset(builder_->createNetworkV2(1U));
  NNADAPTER_CHECK(network_);
  config_.reset(builder_->createBuilderConfig());
  NNADAPTER_CHECK(config_);
  // Convert a NNAdapter model to a nv network
  Converter converter(network_.get(), &tensors_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Mark output
  for (auto& out_operand : model->output_operands) {
    NNADAPTER_CHECK(tensors_.count(out_operand));
    network_->markOutput(*tensors_[out_operand].back());
  }
  // Set config_
  // Identify network
  plan_.reset(builder_->buildSerializedNetwork(*network_, *config_));
  NNADAPTER_CHECK(plan_);
  // Create runtime
  runtime_.reset(nvinfer1::createInferRuntime(*TrtLogger::Global()));
  NNADAPTER_CHECK(runtime_);
  engine_.reset(runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
  NNADAPTER_CHECK(engine_);
  nv_context_.reset(engine_->createExecutionContext());
  NNADAPTER_CHECK(nv_context_);
  // Identify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  input_types_.resize(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto operand = model->input_operands[i];
    const auto& type = operand->type;
    input_types_[i] = type;
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    output_types_[i] = type;
  }
  // malloc gpu memory
  for (int i = 0; i < engine_->getNbBindings(); i++) {
    auto dims = nv_context_->getBindingDimensions(i);
    size_t vol = std::accumulate(
        dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
    size_t type_size = GetNVTypeSize(engine_->getBindingDataType(i));
    std::shared_ptr<void> device_data;
    void* data_ptr;
    cudaMalloc(&data_ptr, type_size * vol);
    device_data.reset(data_ptr, [](void* ptr) { cudaFree(ptr); });
    device_data_.push_back(std::move(device_data));
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  NNADAPTER_LOG(FATAL) << "Build from cache is unimpleted.";
  return NNADAPTER_DEVICE_INTERNAL_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  std::vector<void*> ptrs;
  for (auto& item : device_data_) {
    ptrs.push_back(item.get());
  }
  // Feed inputs
  for (uint32_t i = 0; i < input_count; i++) {
    void* dst_ptr = ptrs[i];
    auto& arg = input_arguments[i];
    auto type = input_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type);
    auto length = GetOperandTypeBufferLength(type);
    NNADAPTER_CHECK_EQ(
        cudaMemcpy(dst_ptr, buffer, length, cudaMemcpyHostToDevice),
        cudaSuccess);
  }
  // Execute model
  nv_context_->execute(1, ptrs.data());
  // Get outputs
  for (uint32_t i = 0; i < output_count; i++) {
    void* src_ptr = ptrs[i + input_count];
    auto& arg = output_arguments[i];
    auto type = &output_types_[arg.index];
    auto buffer = arg.access(arg.memory, type);
    auto length = GetOperandTypeBufferLength(*type);
    NNADAPTER_CHECK_EQ(
        cudaMemcpy(buffer, src_ptr, length, cudaMemcpyDeviceToHost),
        cudaSuccess);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
