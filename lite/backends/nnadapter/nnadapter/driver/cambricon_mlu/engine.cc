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

#include "driver/cambricon_mlu/engine.h"
#include <unistd.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include "driver/cambricon_mlu/converter.h"
#include "driver/cambricon_mlu/optimizer/fix_quantized_ops.h"
#include "driver/cambricon_mlu/optimizer/nchw2nhwc.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the build parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  std::string build_config_file_path;
  auto key_values = GetKeyValues(properties);
  if (key_values.count(CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH)) {
    build_config_file_path = key_values[CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH];
  } else {
    build_config_file_path =
        GetStringFromEnv(CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH);
  }
  if (!build_config_file_path.empty()) {
    build_config_file_path_ = build_config_file_path;
    NNADAPTER_LOG(INFO) << "BuildConfig file path: " << build_config_file_path_;
  } else {
    NNADAPTER_LOG(INFO) << "Build model with default config.";
  }
}

Context::~Context() {}

Program::~Program() {
  Clear();
  if (queue_) {
    MLU_CNRT_CHECK(cnrtDestroyQueue(queue_));
    queue_ = nullptr;
  }
}

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
  dump_graph_path_ = "";
  dump_graph_buffer_ = nullptr;
}

int Program::Build(hal::Model* model, hal::Cache* cache) {
  Clear();
  if (model && cache->dir && cache->token) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->token);
  }
  dump_graph_buffer_ = &cache->buffer;
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(BuildFromModel(model), NNADAPTER_NO_ERROR);
  } else {
    NNADAPTER_CHECK_EQ(BuildFromCache(cache), NNADAPTER_NO_ERROR);
  }
  mm_engine_.reset(mm_model_->CreateIEngine());
  mm_context_.reset(mm_engine_->CreateIContext());
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(hal::Cache* cache) {
  NNADAPTER_LOG(FATAL) << "Build from cache is unimpleted.";
  return NNADAPTER_DEVICE_INTERNAL_ERROR;
}

int Program::BuildFromModel(hal::Model* model) {
  Clear();
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseMatMulAddIntoFullyConnected(model);
  FixQuantizedOps(model);
  ConvertDataLayoutNCHWToNHWC(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  Converter converter(&tensors_, mm_network_.get());
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);

  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<magicmind::ITensor*> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_types_.resize(input_count);
    input_names_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
      input_tensors[i] = tensors_[operand].back();
      NNADAPTER_CHECK(input_tensors[i]);
      input_types_[i] = type;
      input_names_[i] = input_tensors[i]->GetTensorName();
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<magicmind::ITensor*> output_tensors(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    mm_network_->MarkOutput(tensors_[operand].back());
    NNADAPTER_CHECK(output_tensors[i]);
    output_types_[i] = type;
  }
  // Get the info of inputs and outputs, and check the count and buffer size of
  // inputs and outputs
  int num_inputs = mm_network_->GetInputCount();
  NNADAPTER_CHECK_EQ(input_count, num_inputs);
  int num_outputs = mm_network_->GetOutputCount();
  NNADAPTER_CHECK_EQ(output_count, num_outputs);

  magicmind::IBuilderConfig* config = mm_builder_config_.get();
  magicmind::INetwork* network = mm_network_.get();
  if (context_->GetBuildConfigFilePath().empty()) {
    config->ParseFromString(R"({"graph_shape_mutable": true})");
    config->ParseFromString(
        R"({"precision_config": {"precision_mode": "force_float32"}})");
  } else {
    config->ParseFromFile(context_->GetBuildConfigFilePath());
  }
  mm_model_.reset(mm_builder_->BuildModel("camb_model", network, config));
  NNADAPTER_VLOG(3) << "Build success.";
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     hal::Argument* input_arguments,
                     uint32_t output_count,
                     hal::Argument* output_arguments) {
  NNADAPTER_VLOG(3) << "Execute begining.";
  std::vector<magicmind::IRTTensor*> inputs = {};
  std::vector<magicmind::IRTTensor*> outputs = {};
  magicmind::CreateInputTensors(mm_context_.get(), &inputs);
  for (uint32_t i = 0; i < input_count; i++) {
    void* ptr = nullptr;
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &input_types_[arg.index];
    auto buffer = arg.access(arg.memory, type);
    NNADAPTER_CHECK(buffer);
    auto length = GetOperandTypeBufferLength(*type);
    MLU_CNRT_CHECK(cnrtMalloc(&ptr, length));
    MLU_CNRT_CHECK(
        cnrtMemcpy(ptr, buffer, length, CNRT_MEM_TRANS_DIR_HOST2DEV));
    auto tensor_name = input_names_[arg.index];
    auto input_tensor = magicmind::FindIRTTensorByName(inputs, tensor_name);
    input_tensor->SetData(ptr);
    input_tensor->SetDimensions(
        ConvertToMagicMindDims(type->dimensions.data, type->dimensions.count));
  }

  mm_context_->Enqueue(inputs, &outputs, queue_);
  MLU_CNRT_CHECK(cnrtSyncQueue(queue_));
  NNADAPTER_VLOG(3) << "Execute ending.";
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    auto buffer = arg.access(arg.memory, type);
    NNADAPTER_CHECK(buffer);
    void* output_mlu_ptr = outputs[i]->GetMutableData();
    if (IsDeviceMemory(output_mlu_ptr)) {
      MLU_CNRT_CHECK(cnrtMemcpy(buffer,
                                output_mlu_ptr,
                                outputs[i]->GetSize(),
                                CNRT_MEM_TRANS_DIR_DEV2HOST));
    } else {
      memcpy(buffer, output_mlu_ptr, outputs[i]->GetSize());
    }
  }

  for (auto input : inputs) {
    MLU_CNRT_CHECK(cnrtFree(input->GetMutableData()));
    input->Destroy();
  }
  for (auto output : outputs) {
    output->Destroy();
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
