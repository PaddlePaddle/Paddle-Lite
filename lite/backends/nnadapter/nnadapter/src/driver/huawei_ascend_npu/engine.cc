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

#include "driver/huawei_ascend_npu/engine.h"
#include <utility>
#include "driver/huawei_ascend_npu/optimizer/fix_multiple_outputs_ops.h"
#include "driver/huawei_ascend_npu/optimizer/fix_no_inputs_ops.h"
#include "driver/huawei_ascend_npu/optimizer/fix_quantized_ops.h"
#include "driver/huawei_ascend_npu/optimizer/fix_reduce_ops_scalar_output.h"
#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
#include "optimizer/fuse_conv2d_add_into_conv2d.h"
#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "optimizer/fuse_matmul_dequant_add_into_fully_connected_dequant.h"
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

Device::Device() { InitializeAscendCL(); }

Device::~Device() {}

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  // HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS
  std::string selected_device_ids;
  if (key_values.count(HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS)) {
    selected_device_ids = key_values[HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS];
  } else {
    selected_device_ids =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS);
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
  // HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH
  if (key_values.count(HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH)) {
    ascend_config_params_.profiling_file_path =
        key_values[HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH];
  } else {
    ascend_config_params_.profiling_file_path =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH);
  }
  NNADAPTER_LOG(INFO) << "profiling path: "
                      << ascend_config_params_.profiling_file_path;
  // HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH
  if (key_values.count(HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH)) {
    ascend_config_params_.dump_model_path =
        key_values[HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH];
  } else {
    ascend_config_params_.dump_model_path =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH);
  }
  NNADAPTER_LOG(INFO) << "dump model path: "
                      << ascend_config_params_.dump_model_path;
  // HUAWEI_ASCEND_NPU_PRECISION_MODE
  if (key_values.count(HUAWEI_ASCEND_NPU_PRECISION_MODE)) {
    ascend_config_params_.precision_mode =
        key_values[HUAWEI_ASCEND_NPU_PRECISION_MODE];
  } else {
    ascend_config_params_.precision_mode =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_PRECISION_MODE);
  }
  NNADAPTER_LOG(INFO) << "precision mode: "
                      << ascend_config_params_.precision_mode;
  if (ascend_config_params_.precision_mode == "allow_mix_precision") {
    // HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH
    if (key_values.count(HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH)) {
      ascend_config_params_.modify_mixlist_path =
          key_values[HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH];
    } else {
      ascend_config_params_.modify_mixlist_path =
          GetStringFromEnv(HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH);
    }
    NNADAPTER_LOG(INFO) << "modify mixlist path: "
                        << ascend_config_params_.modify_mixlist_path;
  }
  // HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE
  if (key_values.count(HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE)) {
    ascend_config_params_.op_select_impl_mode =
        key_values[HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE];
  } else {
    ascend_config_params_.op_select_impl_mode =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE);
  }
  NNADAPTER_LOG(INFO) << "op select impl mode: "
                      << ascend_config_params_.op_select_impl_mode;
  // HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE
  if (key_values.count(HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE)) {
    ascend_config_params_.op_type_list_for_impl_mode =
        key_values[HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE];
  } else {
    ascend_config_params_.op_type_list_for_impl_mode =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE);
  }
  NNADAPTER_LOG(INFO) << "op type list for impl mode: "
                      << ascend_config_params_.op_type_list_for_impl_mode;
  // HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT
  if (key_values.count(HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT)) {
    ascend_config_params_.enable_compress_weight =
        key_values[HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT];
  } else {
    ascend_config_params_.enable_compress_weight =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT);
  }
  NNADAPTER_LOG(INFO) << "enable compressw weight: "
                      << ascend_config_params_.enable_compress_weight;
  // HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE
  if (key_values.count(HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE)) {
    ascend_config_params_.auto_tune_mode =
        key_values[HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE];
  } else {
    ascend_config_params_.auto_tune_mode =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE);
  }
  NNADAPTER_LOG(INFO) << "auto tune mode: "
                      << ascend_config_params_.auto_tune_mode;
  // HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE
  if (key_values.count(HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE)) {
    ascend_config_params_.enable_dynamic_shape_range =
        key_values[HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE];
  } else {
    ascend_config_params_.enable_dynamic_shape_range =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE);
  }
  NNADAPTER_LOG(INFO) << "enable dynamic shape range: "
                      << ascend_config_params_.enable_dynamic_shape_range;
  // HUAWEI_ASCEND_NPU_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE
  std::string initial_buffer_length_of_dynamic_shape_range;
  if (key_values.count(
          HUAWEI_ASCEND_NPU_INITIAL_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE)) {
    initial_buffer_length_of_dynamic_shape_range = key_values
        [HUAWEI_ASCEND_NPU_INITIAL_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE];
  } else {
    initial_buffer_length_of_dynamic_shape_range = GetStringFromEnv(
        HUAWEI_ASCEND_NPU_INITIAL_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE);
  }
  if (!initial_buffer_length_of_dynamic_shape_range.empty()) {
    ascend_config_params_.initial_buffer_length_of_dynamic_shape_range =
        std::stoll(initial_buffer_length_of_dynamic_shape_range);
  }
  NNADAPTER_LOG(INFO)
      << "initial buffer length of dynamic shape range: "
      << ascend_config_params_.initial_buffer_length_of_dynamic_shape_range;
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  operators_.clear();
  model_client_ = nullptr;
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();

  // Get dynamic_shape_info, optional_shape_str, dynamic_shape_mode_
  if (!cache->buffer.empty()) {
    input_types_ = cache->input_types;
  } else {
    for (auto input_operand : model->input_operands) {
      input_types_.push_back(input_operand->type);
    }
  }
  std::vector<std::string> dynamic_shape_info;
  std::string optional_shape_str;
  if (context_->ascend_config_params()->enable_dynamic_shape_range == "true") {
    dynamic_shape_mode_ = DYNAMIC_SHAPE_MODE_SHAPE_RANGE;
  }
  GetDynamicShapeInfo(input_types_,
                      &dynamic_shape_info,
                      &optional_shape_str,
                      &dynamic_shape_mode_);
  for (auto dynamic_shape : dynamic_shape_info) {
    NNADAPTER_VLOG(3) << "dynamic_shape: " << dynamic_shape;
  }
  NNADAPTER_VLOG(3) << "optional_shape_str: " << optional_shape_str;
  NNADAPTER_VLOG(3) << "dynamic_shape_mode_: " << dynamic_shape_mode_;

  std::vector<uint8_t> model_content;
  std::vector<uint8_t>* model_buffer = nullptr;
  if (!cache->buffer.empty()) {
    // Build from cache
    model_buffer = &cache->buffer;
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    auto output_count = cache->output_types.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_types_ = cache->output_types;
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    FixMultipleOutputsOps(model);
    FixNoInputsOps(model);
    FixReduceOpsScalarOutput(model);
    FuseConv2DBatchNormIntoConv2D(model);
    FuseConv2DAddIntoConv2D(model);
    FuseConv2DActivationIntoConv2D(model);
    FuseMatMulDequantAddIntoFullyConnectedDequant(model);
    FuseMatMulAddIntoFullyConnected(model, true);
    FuseReshapeTransposeReshapeIntoChannelShuffle(model);
    FixQuantizedOps(model);
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
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
        input_operators[i] = *operators_[operand].front()->op();
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
    // Build a GE graph to a CANN OM model, and serialize it into a buffer
    if (!BuildOMModelToBuffer(input_operators,
                              output_operators,
                              model_buffer,
                              dynamic_shape_info,
                              optional_shape_str,
                              dynamic_shape_mode_,
                              context_->ascend_config_params())) {
      NNADAPTER_LOG(FATAL)
          << "Failed to build a CANN OM model and serialize it into a buffer!";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    } else {
      NNADAPTER_VLOG(3)
          << "Build a CANN OM model and serialize it into a buffer success.";
    }
  }
  NNADAPTER_CHECK(model_buffer);
  // Load a CANN OM model from a buffer, and create a CANN model manager
  // client(from CANN service) for inference
  model_client_ = LoadOMModelFromBuffer(*model_buffer,
                                        context_->first_device_id(),
                                        context_->ascend_config_params());
  if (!model_client_) {
    NNADAPTER_LOG(FATAL) << "Failed to load a CANN OM model from a buffer!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Initialize the CANN input and output tensors
  std::vector<ge::TensorDesc> input_tensor_descs, output_tensor_descs;
  if (!model_client_->GetModelIOTensorDim(&input_tensor_descs,
                                          &output_tensor_descs)) {
    NNADAPTER_LOG(FATAL) << "Failed to call GetModelIOTensorDim to get the "
                            "description of input and output tensors!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  auto input_count = input_types_.size();
  if (dynamic_shape_mode_ == DYNAMIC_SHAPE_MODE_NONE) {
    NNADAPTER_CHECK_EQ(input_tensor_descs.size(), input_count);
  } else {
    NNADAPTER_CHECK_GE(input_tensor_descs.size(), input_count);
  }
  for (size_t i = 0; i < input_count; i++) {
    auto type = &input_types_[i];
    auto dimensions = input_tensor_descs[i].GetShape();
    NNADAPTER_VLOG(3) << "CANN input tensors[" << i
                      << "]: " << GEShapeToString(dimensions) << " "
                      << DimensionsToString(type->dimensions.data,
                                            type->dimensions.count);
    NNADAPTER_CHECK_EQ(dimensions.GetDimNum(), type->dimensions.count);
    for (size_t j = 0; j < type->dimensions.count; j++) {
      auto dimension = type->dimensions.data[j];
      if (dimension == NNADAPTER_UNKNOWN) {
        // Check if the dimension of the model inputs is dynamic
        NNADAPTER_CHECK_EQ(dimensions.GetDim(j), -1)
            << "The " << j << "th dimension of the " << i
            << "th input does not match, expect " << dimension
            << " but recevied " << dimensions.GetDim(j);
      }
    }
  }
  auto output_count = output_types_.size();
  NNADAPTER_CHECK_EQ(output_tensor_descs.size(), output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto dimensions = output_tensor_descs[i].GetShape();
    NNADAPTER_VLOG(3) << "CANN output tensors[" << i
                      << "]: " << GEShapeToString(dimensions) << " "
                      << DimensionsToString(type->dimensions.data,
                                            type->dimensions.count);
    NNADAPTER_CHECK_EQ(dimensions.GetDimNum(), type->dimensions.count);
    for (size_t j = 0; j < type->dimensions.count; j++) {
      auto dimension = type->dimensions.data[j];
      if (dimension > 0) {
        // Check if the dimension of the model outputs is not dynamic
        NNADAPTER_CHECK_EQ(dimension, dimensions.GetDim(j))
            << "The " << j << "th dimension of the " << i
            << "th input does not match, expect " << dimension
            << " but recevied " << dimensions.GetDim(j);
      }
    }
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
    bool is_matched = true;
    for (uint32_t j = 0; j < count; j++) {
      if (data[j] != src_data[j]) {
        is_matched = false;
        break;
      }
    }
    if (is_matched) continue;
    // Check dynamic dymensions data
    for (uint32_t k = 0; k < src_dimensions.dynamic_count; k++) {
      is_matched = true;
      for (uint32_t j = 0; j < count; j++) {
        if (data[j] != src_dimensions.dynamic_data[k][j] &&
            src_dimensions.dynamic_data[k][j] != -1) {
          is_matched = false;
          break;
        }
      }
      if (is_matched) break;
    }
    if (!is_matched) {
      return NNADAPTER_INVALID_DIMENSIONS;
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
  NNADAPTER_CHECK(model_client_->Process(input_count,
                                         &input_types_,
                                         input_arguments,
                                         output_count,
                                         &output_types_,
                                         output_arguments,
                                         dynamic_shape_mode_));
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
