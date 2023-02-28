// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/intel_openvino/engine.h"
#include "driver/intel_openvino/converter/converter.h"
#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
#include "optimizer/fuse_conv2d_add_into_conv2d.h"
#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace intel_openvino {

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  std::string selected_device_names;
  if (key_values.count(INTEL_OPENVINO_SELECT_DEVICE_NAMES)) {
    selected_device_names = key_values[INTEL_OPENVINO_SELECT_DEVICE_NAMES];
  } else {
    selected_device_names =
        GetStringFromEnv(INTEL_OPENVINO_SELECT_DEVICE_NAMES);
  }
  if (!selected_device_names.empty()) {
    selected_device_names_ =
        string_split<std::string>(selected_device_names, ",");
  }
  NNADAPTER_CHECK_GE(selected_device_names_.size(), 1);
  // Only supports specifying one device
  if (selected_device_names_.size() > 1) {
    NNADAPTER_LOG(WARNING) << "Only supports specifying one device, so the "
                              "first one is selected and others will be "
                              "ignored.";
    auto first_device_name = selected_device_names_[0];
    selected_device_names_.clear();
    selected_device_names_.push_back(first_device_name);
  }
  NNADAPTER_LOG(INFO) << "selected device types: ";
  for (auto& selected_device_name : selected_device_names_) {
    NNADAPTER_LOG(INFO) << selected_device_name;
  }
  // INTEL_OPENVINO_INFERENCE_NUM_THREADS.
  int inference_threads_num = -1;
  if (key_values.count(INTEL_OPENVINO_INFERENCE_NUM_THREADS)) {
    inference_threads_num =
        std::stoi(key_values[INTEL_OPENVINO_INFERENCE_NUM_THREADS]);
  } else {
    auto thread_num_str =
        GetStringFromEnv(INTEL_OPENVINO_INFERENCE_NUM_THREADS);
    if (!thread_num_str.empty()) {
      inference_threads_num = std::stoi(thread_num_str);
    }
  }
  auto& device_config = device_config_map_[GetFirtSelectedDeviceName()];
  if (inference_threads_num >= 0) {
    device_config.emplace(ov::inference_num_threads(inference_threads_num));
  }
  NNADAPTER_LOG(INFO)
      << "Maximum number of threads that can be used for inference tasks: "
      << inference_threads_num;
}

Context::~Context() {}

int Program::Build(core::Model* model, core::Cache* cache) {
  NNADAPTER_LOG(INFO) << "OpenVINO runtime version - "
                      << ov::get_openvino_version();
  runtime_core_ = std::make_shared<ov::Core>();
  auto device_name = context_->GetFirtSelectedDeviceName();
  NNADAPTER_LOG(INFO) << device_name << " version - "
                      << runtime_core_->get_versions(device_name);
  InitializeDeviceConfig(
      device_name, runtime_core_, context_->GetDeviceConfig());
  if (!cache->buffer.empty()) {
    // Build from cache
    input_types_ = cache->input_types;
    output_types_ = cache->output_types;
    // Check if the cached model is dynamic shape.
    for (auto type : input_types_) {
      if (IsDynamicShapeOperandType(type)) {
        with_dynamic_shape_ = true;
        break;
      }
    }
    std::stringstream model_stream;
    model_stream.write(reinterpret_cast<char*>(cache->buffer.data()),
                       cache->buffer.size());
    NNADAPTER_VLOG(3) << "NNAdapter model cache size(bytes):"
                      << model_stream.str().size();
    compiled_model_ =
        std::make_shared<ov::CompiledModel>(runtime_core_->import_model(
            model_stream, context_->GetFirtSelectedDeviceName()));
    NNADAPTER_VLOG(3) << "Build from cache success.";
  } else {
    // Build from model
    NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
    FuseConv2DBatchNormIntoConv2D(model);
    FuseConv2DAddIntoConv2D(model);
    FuseConv2DActivationIntoConv2D(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
    Converter converter(&parameter_node_map_, &tensor_map_);
    NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
    // Indentify the inputs and outputs
    auto input_count = model->input_operands.size();
    auto parameter_nodes =
        std::vector<std::shared_ptr<default_opset::Parameter>>();
    NNADAPTER_VLOG(3) << "Model input count: " << input_count;
    NNADAPTER_CHECK_EQ(input_count, parameter_node_map_.size());
    if (input_count > 0) {
      input_types_.resize(input_count);
      for (size_t i = 0; i < input_count; i++) {
        auto operand = model->input_operands[i];
        const auto& type = operand->type;
        NNADAPTER_CHECK(tensor_map_.find(operand) != tensor_map_.end());
        input_types_[i] = type;
        NNADAPTER_CHECK(parameter_node_map_.find(operand) !=
                        parameter_node_map_.end());
        parameter_nodes.push_back(parameter_node_map_[operand]);
      }
    }
    auto output_count = model->output_operands.size();
    NNADAPTER_VLOG(3) << "Model output count: " << output_count;
    NNADAPTER_CHECK_GT(output_count, 0);
    output_types_.resize(output_count);
    for (size_t i = 0; i < output_count; i++) {
      auto operand = model->output_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensor_map_.find(operand) != tensor_map_.end());
      output_types_[i] = type;
      auto result_node =
          std::make_shared<default_opset::Result>(*tensor_map_[operand].back());
      result_nodes_.push_back(result_node);
    }
    // Get dynamic shape info.
    std::map<ov::Output<ov::Node>, ov::PartialShape> dynamic_shape_map;
    for (auto operand : model->input_operands) {
      bool dynamic_shape_operand = false;
      int count = operand->type.dimensions.count;
      auto dims = operand->type.dimensions.data;
      for (int i = 0; i < count; i++) {
        if (dims[i] == NNADAPTER_UNKNOWN) {
          with_dynamic_shape_ = true;
          dynamic_shape_operand = true;
          break;
        }
      }
      if (dynamic_shape_operand) {
        ov::PartialShape dynamic_shape =
            ConvertDynamicDimensions(&operand->type.dimensions);
        dynamic_shape_map[parameter_node_map_[operand]->output(0)] =
            dynamic_shape;
      }
    }
    // Convert NNAdapter model to OpenVINO model
    std::shared_ptr<ov::Model> ov_model = std::make_shared<ov::Model>(
        result_nodes_, parameter_nodes, "openvino_graph");
    // Set dynamic shape for OpenVino model.
    if (with_dynamic_shape_) {
      ov_model->reshape(dynamic_shape_map);
    }
    // Cache compiled openvino model.
    if (cache->token && cache->dir) {
      compiled_model_ = std::make_shared<ov::CompiledModel>(
          runtime_core_->compile_model(ov_model,
                                       context_->GetFirtSelectedDeviceName(),
                                       ov::cache_dir(cache->dir)));
      std::stringstream model_stream;
      compiled_model_->export_model(model_stream);
      NNADAPTER_VLOG(3) << "NNAdapter model cache size(bytes):"
                        << model_stream.str().size();
      cache->buffer.resize(model_stream.str().size());
      memcpy(reinterpret_cast<char*>(cache->buffer.data()),
             model_stream.str().data(),
             model_stream.str().size());
    } else {
      compiled_model_ =
          std::make_shared<ov::CompiledModel>(runtime_core_->compile_model(
              ov_model, context_->GetFirtSelectedDeviceName()));
    }
    NNADAPTER_VLOG(3) << "Build from model success.";
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
    if (with_dynamic_shape_) {
      for (int i = 0; i < count; i++) {
        is_matched = false;
        int min_shape = src_dimensions.dynamic_data[0][i];
        int max_shape = src_dimensions.dynamic_data[0][i];
        for (int j = 0; j < src_dimensions.dynamic_count; j++) {
          int shape = src_dimensions.dynamic_data[j][i];
          if (shape == -1) {
            is_matched = true;
            break;
          }
          if (shape < min_shape) {
            min_shape = shape;
          }
          if (shape > max_shape) {
            max_shape = shape;
          }
        }
        if (is_matched) continue;
        if (data[i] < min_shape || data[i] > max_shape) {
          NNADAPTER_VLOG(5) << "NNAdapter invalid dimensions:" << data[i]
                            << " not in[" << min_shape << "," << max_shape
                            << "]";
          return NNADAPTER_INVALID_DIMENSIONS;
        }
      }
    } else {
      NNADAPTER_VLOG(5) << "NNAdapter invalid dimensions.";
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
  NNADAPTER_CHECK(compiled_model_);
  // Create infer request
  ov::InferRequest infer_request = compiled_model_->create_infer_request();
  // Set inputs
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
    ov::Tensor input_tensor = infer_request.get_input_tensor(i);
    if (with_dynamic_shape_) {
      input_tensor.set_shape(ConvertToOVShape(type.dimensions));
    }
    memcpy(input_tensor.data(), buffer, length);
  }
  // Inference
  infer_request.infer();
  // Get results
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    ov::Tensor output_tensor = infer_request.get_output_tensor(i);
    auto output_size = output_tensor.get_byte_size();
    // Skip copy to output if output tensor if empty.
    if (output_size == 0) {
      continue;
    }
    auto ov_out_shape = output_tensor.get_shape();
    // Get the dimensions of the outputs from OpenVINO
    // according to the dynamic dimensions of the inputs, fill them to 'type'
    // and call the 'access' function to re-allocate the host output memory.
    auto type = output_types_[arg.index];
    if (with_dynamic_shape_ || IsDynamicShapeOperandType(type)) {
      NNADAPTER_CHECK_EQ(ov_out_shape.size(), type.dimensions.count);
      for (int i = 0; i < type.dimensions.count; i++) {
        type.dimensions.data[i] = ov_out_shape[i];
      }
    }
    auto buffer = arg.access(arg.memory, &type, nullptr);
    auto length = GetOperandTypeBufferLength(type);
    NNADAPTER_CHECK_EQ(output_size, length);
    memcpy(buffer, output_tensor.data(), output_size);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
