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
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

static void UpdateDynamicShapeMode(
    const NNAdapterOperandDimensionType& dimensions, DynamicShapeMode* mode) {
  bool is_nchw = dimensions.count == 4;
  bool b_unk = dimensions.data[0] == NNADAPTER_UNKNOWN;
  bool c_unk = dimensions.data[1] == NNADAPTER_UNKNOWN;
  bool h_unk = dimensions.data[2] == NNADAPTER_UNKNOWN;
  bool w_unk = dimensions.data[3] == NNADAPTER_UNKNOWN;
  if (is_nchw && b_unk && !c_unk && !h_unk && !w_unk) {
    if (*mode == DYNAMIC_SHAPE_MODE_NONE) {
      *mode = DYNAMIC_SHAPE_MODE_BTACH_SIZE;
    }
    if (*mode != DYNAMIC_SHAPE_MODE_BTACH_SIZE) {
      *mode = DYNAMIC_SHAPE_MODE_N_DIMS;
    }
  } else if (is_nchw && !b_unk && !c_unk && (h_unk || w_unk)) {
    if (*mode == DYNAMIC_SHAPE_MODE_NONE) {
      *mode = DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH;
    } else {
      // only support one input has dynamic h&w
      *mode = DYNAMIC_SHAPE_MODE_N_DIMS;
    }
  } else {
    *mode = DYNAMIC_SHAPE_MODE_N_DIMS;
  }
}

static void GetDynamicInfo(const std::vector<NNAdapterOperandType>& input_types,
                           std::vector<std::string>* shapes,
                           std::string* optional_shapes_str,
                           DynamicShapeMode* mode) {
  // Get dynamic shape mode from all inputs. Rules are as follows:
  // 1. If all shapes are const, mode is DYNAMIC_SHAPE_MODE_NONE.
  // 2. If only batch of inputs is unknown, mode is
  // DYNAMIC_SHAPE_MODE_BTACH_SIZE.
  // 3. If only one 4-D input has dynamic height or weight, mode is
  // DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH.
  // 4. Others belong to DYNAMIC_SHAPE_MODE_N_DIMS.
  *mode = DYNAMIC_SHAPE_MODE_NONE;
  for (auto& input_type : input_types) {
    auto dimensions = input_type.dimensions;
    if (dimensions.dynamic_count == 0) continue;
    UpdateDynamicShapeMode(dimensions, mode);
  }

  // Generate shapes string according to mode.
  std::vector<std::string> optional_shapes;
  for (auto& input_type : input_types) {
    auto dimensions = input_type.dimensions;
    if (dimensions.dynamic_count == 0) {
      std::vector<int32_t> shape(dimensions.data,
                                 dimensions.data + dimensions.count);
      shapes->push_back(ShapeToString(shape));
      continue;
    }

    if (optional_shapes.empty()) {
      optional_shapes.resize(dimensions.dynamic_count);
    }

    std::vector<int32_t> shape(dimensions.data,
                               dimensions.data + dimensions.count);
    switch (*mode) {
      case DYNAMIC_SHAPE_MODE_BTACH_SIZE: {
        shapes->push_back(ShapeToString(shape));
        for (size_t i = 0; i < optional_shapes.size(); i++) {
          auto& optional_shape_str = optional_shapes.at(i);
          auto dynamic_batch_str =
              std::to_string(dimensions.dynamic_data[i][0]);
          if (optional_shape_str.empty()) {
            optional_shape_str = dynamic_batch_str;
          }
          NNADAPTER_CHECK_EQ(optional_shape_str, dynamic_batch_str);
        }
      } break;
      case DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH: {
        NNADAPTER_CHECK_EQ(shape.size(), 4UL);
        shape[2] = -1;
        shape[3] = -1;
        shapes->push_back(ShapeToString(shape));
        for (size_t i = 0; i < optional_shapes.size(); i++) {
          auto& optional_shape_str = optional_shapes.at(i);
          NNADAPTER_CHECK(optional_shape_str.empty());
          optional_shape_str = std::to_string(dimensions.dynamic_data[i][2]) +
                               "," +
                               std::to_string(dimensions.dynamic_data[i][3]);
        }
      } break;
      case DYNAMIC_SHAPE_MODE_N_DIMS: {
        shapes->push_back(ShapeToString(shape));
        for (size_t i = 0; i < optional_shapes.size(); i++) {
          auto& optional_shape_str = optional_shapes.at(i);
          for (uint32_t j = 0; j < dimensions.count; j++) {
            if (dimensions.data[j] != NNADAPTER_UNKNOWN) continue;
            if (!optional_shape_str.empty()) {
              optional_shape_str += ",";
            }
            optional_shape_str += dimensions.dynamic_data[i][j];
          }
        }
      } break;
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported dynamic shape mode: " << mode;
        break;
    }
  }

  *optional_shapes_str = MergeOptionalShapesString(optional_shapes, *mode);
}

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
    profiling_file_path_ = key_values[HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH];
  } else {
    profiling_file_path_ =
        GetStringFromEnv(HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH);
  }
  NNADAPTER_LOG(INFO) << "profiling path: " << profiling_file_path_;
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  operators_.clear();
  model_client_ = nullptr;
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();

  std::vector<std::string> dynamic_shapes;
  std::string optional_shapes_str;
  std::vector<NNAdapterOperandType> input_types;
  if (!cache->buffer.empty()) {
    input_types = cache->input_types;
  } else {
    for (auto input_operand : model->input_operands) {
      input_types.push_back(input_operand->type);
    }
  }
  GetDynamicInfo(
      input_types, &dynamic_shapes, &optional_shapes_str, &dynamic_shape_mode_);
  for (auto dynamic_shape : dynamic_shapes) {
    NNADAPTER_VLOG(3) << "dynamic_shape: " << dynamic_shape;
  }
  NNADAPTER_VLOG(3) << "optional_shapes_str: " << optional_shapes_str;
  NNADAPTER_VLOG(3) << "dynamic_shape_mode_: " << dynamic_shape_mode_;

  std::vector<uint8_t> model_content;
  std::vector<uint8_t>* model_buffer = nullptr;
  if (!cache->buffer.empty()) {
    // Build from cache
    model_buffer = &cache->buffer;
    auto input_count = cache->input_types.size();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    input_types_ = cache->input_types;
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
    FuseMatMulAddIntoFullyConnected(model);
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
      input_types_.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        NNADAPTER_CHECK(operators_.find(operand) != operators_.end());
        input_operators[i] = *operators_[operand].back()->op();
        input_types_[i] = operand->type;
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
                              dynamic_shapes,
                              optional_shapes_str,
                              dynamic_shape_mode_)) {
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
                                        context_->GetFirstDeviceID(),
                                        context_->GetProfilingFilePath());
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
      if (dimension == -1) {
        // Check if the dimension of the model inputs is dynamic
        NNADAPTER_CHECK_EQ(dimension, dimensions.GetDim(j))
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
      if (dimension != -1) {
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

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
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
