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
  } else {
    selected_device_names_ = std::vector<std::string>({0});
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
}

Context::~Context() {}

int Program::Build(core::Model* model, core::Cache* cache) {
  NNADAPTER_LOG(INFO) << "OpenVINO runtime version - "
                      << ov::get_openvino_version();
  runtime_core_ = std::make_shared<ov::Core>();
  auto device_name = context_->GetFirtSelectedDeviceName();
  NNADAPTER_LOG(INFO) << device_name << " version - "
                      << runtime_core_->get_versions(device_name);
  return cache->buffer.empty() ? BuildFromModel(model) : BuildFromCache(cache);
}

int Program::BuildFromCache(core::Cache* cache) {
  NNADAPTER_LOG(FATAL) << "Build from cache is unimpleted.";
  return NNADAPTER_DEVICE_INTERNAL_ERROR;
}

int Program::BuildFromModel(core::Model* model) {
  NNADAPTER_VLOG(5) << "NNAdapter model:" << std::endl << Visualize(model);
  Converter converter(&parameter_nodes_, &tensor_map_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  NNADAPTER_CHECK_EQ(input_count, parameter_nodes_.size());
  if (input_count > 0) {
    input_types_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensor_map_.find(operand) != tensor_map_.end());
      input_types_[i] = type;
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
  // Convert NNAdapter model to OpenVINO model
  std::shared_ptr<ov::Model> ov_model = std::make_shared<ov::Model>(
      result_nodes_, parameter_nodes_, "openvino_graph");
  compiled_model_ =
      std::make_shared<ov::CompiledModel>(runtime_core_->compile_model(
          ov_model, context_->GetFirtSelectedDeviceName()));
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
    arg.access(arg.memory, &type);
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
    auto buffer = arg.access(arg.memory, &type);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(type);
    ov::Tensor input_tensor = infer_request.get_input_tensor(i);
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
    auto type = output_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type);
    auto length = GetOperandTypeBufferLength(type);
    ov::Tensor output_tensor = infer_request.get_output_tensor(i);
    auto output_size = output_tensor.get_byte_size();
    NNADAPTER_CHECK_EQ(output_size, length);
    memcpy(buffer, output_tensor.data(), output_size);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
