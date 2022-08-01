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

#include "driver/google_xnnpack/engine.h"
#include <algorithm>
#include <utility>
#include "driver/google_xnnpack/converter/converter.h"
#include "driver/google_xnnpack/converter/validator.h"
#include "driver/google_xnnpack/optimizer/resolve_operation_liminations.h"
#include "optimizer/convert_datalayout_nchw_to_nhwc.h"
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
namespace google_xnnpack {

Device::Device() {
  NNADAPTER_CHECK(xnn_initialize(nullptr) == xnn_status_success)
      << "Failed to initialize XNNPACK library!";
}

Device::~Device() {
  NNADAPTER_CHECK(xnn_deinitialize() == xnn_status_success)
      << "Failed to deinitialize XNNPACK library!";
}

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  std::string key_value;
  auto key_values = GetKeyValues(properties);
  // GOOGLE_XNNPACK_NUM_THREADS
  if (key_values.count(GOOGLE_XNNPACK_NUM_THREADS)) {
    num_threads_ = string_parse<int>(key_values[GOOGLE_XNNPACK_NUM_THREADS]);
  } else {
    num_threads_ = GetIntFromEnv(GOOGLE_XNNPACK_NUM_THREADS, 0);
  }
  NNADAPTER_LOG(INFO) << "num_threads: " << num_threads_;
  if (num_threads_ > 1) {
    threadpool_ = pthreadpool_create(num_threads_);
    NNADAPTER_CHECK(threadpool_ != nullptr)
        << "Failed to create a thread pool for XNNPACK library!";
  }
}

Context::~Context() {
  if (threadpool_) {
    pthreadpool_destroy(threadpool_);
    threadpool_ = nullptr;
  }
}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensor_value_ids_.clear();
  if (runtime_) {
    xnn_delete_runtime(runtime_);
    runtime_ = nullptr;
  }
  if (subgraph_) {
    xnn_delete_subgraph(subgraph_);
    subgraph_ = nullptr;
  }
  input_types_.clear();
  output_types_.clear();
  external_values_.clear();
}

int Program::Validate(const core::Model* model, bool* supported_operations) {
  Validator validator(context_);
  return validator.Apply(model, supported_operations);
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  bool model_from_cache = false;
  if (!cache->buffer.empty()) {
    // Build from cache
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    input_types_ = cache->input_types;
    auto output_count = cache->output_types.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_types_ = cache->output_types;
    NNADAPTER_CHECK(!model);
    if (!DeserializeModel(cache->buffer.data(), cache->buffer.size(), &model)) {
      NNADAPTER_LOG(FATAL)
          << "Failed to deserialize the optimized core::Model from a buffer!";
    } else {
      model_from_cache = true;
      NNADAPTER_VLOG(3)
          << "Deserialize the optimized core::Model from a buffer success.";
    }
    NNADAPTER_VLOG(5) << "Cached model:" << std::endl << Visualize(model);
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    // Convert the data layout and the quantization parameters of the NNAdapter
    // Model
    FuseConv2DBatchNormIntoConv2D(model);
    FuseConv2DAddIntoConv2D(model);
    FuseConv2DActivationIntoConv2D(model);
    FuseMatMulAddIntoFullyConnected(model);
    FuseReshapeTransposeReshapeIntoChannelShuffle(model);
    ResolveOperationLiminations(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  }
  // Convert the NNAdapter model to XNNPACK model
  xnn_status result = xnn_create_subgraph(0, 0, &subgraph_);
  if (result != xnn_status_success) {
    NNADAPTER_LOG(FATAL) << "Failed to create a XNNPACK subgraph(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  Converter converter(subgraph_, &tensor_value_ids_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  if (input_count > 0) {
    input_types_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      NNADAPTER_CHECK(tensor_value_ids_.find(operand) !=
                      tensor_value_ids_.end())
          << "No XNNPACK tensor value id found for input operand @0x"
          << std::hex << reinterpret_cast<int64_t>(operand);
      auto tensor_value_id = tensor_value_ids_[operand].front();
      NNADAPTER_CHECK_NE(tensor_value_id, XNN_INVALID_VALUE_ID);
      NNADAPTER_VLOG(5) << "Found a XNNPACK tensor value id " << tensor_value_id
                        << " for input operand @0x" << std::hex
                        << reinterpret_cast<int64_t>(operand);
      xnn_external_value input_external_value;
      memset(&input_external_value, 0, sizeof(xnn_external_value));
      input_external_value.id = tensor_value_id;
      input_external_value.data =
          nullptr;  // The data ptr will be set at the execution time
      external_values_.push_back(input_external_value);
      input_types_[i] = operand->type;
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(tensor_value_ids_.find(operand) != tensor_value_ids_.end())
        << "No XNNPACK tensor value id found for output operand @0x" << std::hex
        << reinterpret_cast<int64_t>(operand);
    auto tensor_value_id = tensor_value_ids_[operand].back();
    NNADAPTER_CHECK_NE(tensor_value_id, XNN_INVALID_VALUE_ID);
    NNADAPTER_VLOG(5) << "Found a XNNPACK tensor value id " << tensor_value_id
                      << " for output operand @0x" << std::hex
                      << reinterpret_cast<int64_t>(operand);
    xnn_external_value output_external_value;
    memset(&output_external_value, 0, sizeof(xnn_external_value));
    output_external_value.id = tensor_value_id;
    output_external_value.data =
        nullptr;  // The data ptr will be set at the execution time
    external_values_.push_back(output_external_value);
    output_types_[i] = operand->type;
  }
  if (cache->token && cache->dir) {
    // Serialize core::Model to buffer if cache mode is enabled
    if (cache->buffer.empty()) {
      if (!SerializeModel(model, &cache->buffer)) {
        NNADAPTER_LOG(FATAL)
            << "Failed to serialize the optimized core::Model into a buffer!";
      } else {
        NNADAPTER_VLOG(3)
            << "Serialize the optimized core::Model into a buffer success.";
      }
    }
  }
  result =
      xnn_create_runtime_v2(subgraph_, context_->threadpool(), 0, &runtime_);
  if (result != xnn_status_success) {
    NNADAPTER_LOG(FATAL) << "Failed to create a XNNPACK runtime(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Release the restored core::Model
  if (model_from_cache) {
    nnadapter::ClearModel(model);
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
  int result = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (result != NNADAPTER_NO_ERROR) return result;
  // Set inputs and outputs and transform the data with zero point
  for (uint32_t i = 0; i < input_count; i++) {
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = input_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    // auto length = GetOperandTypeBufferLength(type);
    external_values_[arg.index].data = buffer;
  }
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    // TODO(hong19860320) Get the dimensions of the outputs from NNAPI
    // according to the dynamic dimensions of the inputs, fill them to 'type'
    // and call the 'access' function to re-allocate the host output memory
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    // auto length = GetOperandTypeBufferLength(*type);
    external_values_[arg.index + input_count].data = buffer;
  }
  auto start_time = GetCurrentUS();
  NNADAPTER_CHECK(xnn_setup_runtime(runtime_,
                                    external_values_.size(),
                                    external_values_.data()) ==
                  xnn_status_success)
      << "Failed to setup XNNPACK runtime!";
  NNADAPTER_CHECK(xnn_invoke_runtime(runtime_) == xnn_status_success)
      << "Failed to invoke XNNPACK runtime!";
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  return NNADAPTER_NO_ERROR;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
