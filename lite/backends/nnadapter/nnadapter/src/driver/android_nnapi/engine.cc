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

#include "driver/android_nnapi/engine.h"
#include <algorithm>
#include <utility>
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "driver/android_nnapi/optimizer/resolve_operation_liminations.h"
#include "driver/android_nnapi/optimizer/restrict_input_output_quant_params.h"
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
namespace android_nnapi {

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  std::string key_value;
  auto key_values = GetKeyValues(properties);
  NNADAPTER_LOG(INFO) << "Runtime information: " << std::endl
                      << "  Found NNAPI: " << nnapi()->nnapi_exists << std::endl
                      << "  NNAPI runtime feature level: "
                      << nnapi()->nnapi_runtime_feature_level << std::endl
                      << "  Android sdk version: "
                      << nnapi()->android_sdk_version;
  NNADAPTER_CHECK(nnapi()->nnapi_exists) << "NNAPI is not found!";
  // Get the available device list
  if (nnapi()->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_3) {
    std::vector<std::pair<ANeuralNetworksDevice*, std::pair<std::string, int>>>
        available_device_list;
    uint32_t num_devices = 0;
    nnapi()->ANeuralNetworks_getDeviceCount(&num_devices);
    NNADAPTER_LOG(INFO) << "Available devices: ";
    for (uint32_t i = 0; i < num_devices; i++) {
      ANeuralNetworksDevice* device = nullptr;
      const char* name = nullptr;
      const char* version = nullptr;
      int64_t feature_level = 0;
      int32_t type = 0;
      nnapi()->ANeuralNetworks_getDevice(i, &device);
      nnapi()->ANeuralNetworksDevice_getName(device, &name);
      nnapi()->ANeuralNetworksDevice_getVersion(device, &version);
      nnapi()->ANeuralNetworksDevice_getFeatureLevel(device, &feature_level);
      nnapi()->ANeuralNetworksDevice_getType(device, &type);
      available_device_list.emplace_back(
          device, std::pair<std::string, int>(name, type));
      NNADAPTER_LOG(INFO) << "[" << i << "] name: " << name
                          << " version: " << version
                          << " feature level: " << feature_level
                          << " type: " << type;
    }
    // ANDROID_NNAPI_ONLY_USE_ACC_DEVICE
    if (key_values.count(ANDROID_NNAPI_ONLY_USE_ACC_DEVICE)) {
      only_use_acc_device_ =
          string_parse<bool>(key_values[ANDROID_NNAPI_ONLY_USE_ACC_DEVICE]);
    } else {
      only_use_acc_device_ =
          GetBoolFromEnv(ANDROID_NNAPI_ONLY_USE_ACC_DEVICE, false);
    }
    NNADAPTER_LOG(INFO) << "only_use_acc_device: " << only_use_acc_device_;
    // ANDROID_NNAPI_DISABLE_CPU_DEVICE
    if (key_values.count(ANDROID_NNAPI_DISABLE_CPU_DEVICE)) {
      disable_cpu_device_ =
          string_parse<bool>(key_values[ANDROID_NNAPI_DISABLE_CPU_DEVICE]);
    } else {
      disable_cpu_device_ =
          GetBoolFromEnv(ANDROID_NNAPI_DISABLE_CPU_DEVICE, false);
    }
    NNADAPTER_LOG(INFO) << "disable_cpu_device: " << disable_cpu_device_;
    // ANDROID_NNAPI_SELECTED_DEVICE_NAMES
    if (key_values.count(ANDROID_NNAPI_SELECTED_DEVICE_NAMES)) {
      key_value = key_values[ANDROID_NNAPI_SELECTED_DEVICE_NAMES];
    } else {
      key_value = GetStringFromEnv(ANDROID_NNAPI_SELECTED_DEVICE_NAMES);
    }
    if (!key_value.empty()) {
      auto selected_device_names = string_split<std::string>(key_value, ",");
      for (size_t i = 0; i < selected_device_names.size(); i++) {
        bool found = false;
        auto& selected_device_name = selected_device_names[i];
        for (const auto& available_device_info : available_device_list) {
          if (selected_device_name != available_device_info.second.first)
            continue;
          auto selected_device_type = available_device_info.second.second;
          if (selected_device_type == ANEURALNETWORKS_DEVICE_CPU &&
              (only_use_acc_device_ || disable_cpu_device_)) {
            NNADAPTER_LOG(WARNING)
                << selected_device_name
                << " is a cpu device which is in "
                   "ANDROID_NNAPI_SELECTED_DEVICE_NAMES conflicts with "
                   "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE=TRUE or "
                   "ANDROID_NNAPI_DISABLE_CPU_DEVICE=TRUE, "
                   "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE and "
                   "ANDROID_NNAPI_DISABLE_CPU_DEVICE have higher priority, so "
                << selected_device_name
                << " is removed from the selected device list.";
          } else if (selected_device_type !=
                         ANEURALNETWORKS_DEVICE_ACCELERATOR &&
                     only_use_acc_device_) {
            NNADAPTER_LOG(WARNING)
                << selected_device_name
                << " is not a accelerator device which is in "
                   "ANDROID_NNAPI_SELECTED_DEVICE_NAMES conflicts with "
                   "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE=TRUE, "
                   "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE have higher priority, so "
                << selected_device_name
                << " is removed from the selected device list.";
          } else {
            NNADAPTER_LOG(WARNING) << "Add " << selected_device_name;
            selected_devices_.push_back(available_device_info.first);
          }
          found = true;
          break;
        }
        if (!found) {
          NNADAPTER_LOG(WARNING) << selected_device_name << " is not found.";
        }
      }
    }
    // ANDROID_NNAPI_SELECTED_DEVICE_IDS
    if (key_values.count(ANDROID_NNAPI_SELECTED_DEVICE_IDS)) {
      key_value = key_values[ANDROID_NNAPI_SELECTED_DEVICE_IDS];
    } else {
      key_value = GetStringFromEnv(ANDROID_NNAPI_SELECTED_DEVICE_IDS);
    }
    if (selected_devices_.empty() && !key_value.empty()) {
      auto selected_device_ids = string_split<int>(key_value, ",");
      for (size_t i = 0; i < selected_device_ids.size(); i++) {
        auto& selected_device_id = selected_device_ids[i];
        if (selected_device_id < 0 ||
            selected_device_id >= available_device_list.size()) {
          NNADAPTER_LOG(WARNING) << "Invalid device id(" << selected_device_id
                                 << "), out of range[0, "
                                 << available_device_list.size() - 1 << "]";
          continue;
        }
        auto& selected_device_info = available_device_list[selected_device_id];
        auto selected_device_name = selected_device_info.second.first;
        auto selected_device_type = selected_device_info.second.second;
        if (selected_device_type == ANEURALNETWORKS_DEVICE_CPU &&
            (only_use_acc_device_ || disable_cpu_device_)) {
          NNADAPTER_LOG(WARNING)
              << selected_device_name
              << " is a cpu device which is in "
                 "ANDROID_NNAPI_SELECTED_DEVICE_IDS conflicts with "
                 "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE=TRUE or "
                 "ANDROID_NNAPI_DISABLE_CPU_DEVICE=TRUE, "
                 "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE and "
                 "ANDROID_NNAPI_DISABLE_CPU_DEVICE have higher priority, so "
              << selected_device_name
              << " is removed from the selected device list.";
        } else if (selected_device_type != ANEURALNETWORKS_DEVICE_ACCELERATOR &&
                   only_use_acc_device_) {
          NNADAPTER_LOG(WARNING)
              << selected_device_name
              << " is not a accelerator device which is in "
                 "ANDROID_NNAPI_SELECTED_DEVICE_IDS conflicts with "
                 "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE=TRUE, "
                 "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE have higher priority, so "
              << selected_device_name
              << " is removed from the selected device list.";
        } else {
          NNADAPTER_LOG(WARNING) << "Add " << selected_device_name;
          selected_devices_.push_back(selected_device_info.first);
        }
      }
    }
    if (selected_devices_.empty() &&
        (only_use_acc_device_ || disable_cpu_device_)) {
      for (const auto& available_device_info : available_device_list) {
        auto available_device_name = available_device_info.second.first;
        auto available_device_type = available_device_info.second.second;
        if (available_device_type == ANEURALNETWORKS_DEVICE_CPU &&
            (only_use_acc_device_ || disable_cpu_device_)) {
          NNADAPTER_LOG(WARNING)
              << available_device_name
              << " is removed from the selected device list because "
                 "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE=TRUE and "
                 "ANDROID_NNAPI_DISABLE_CPU_DEVICE=TRUE";
        } else if (available_device_type !=
                       ANEURALNETWORKS_DEVICE_ACCELERATOR &&
                   only_use_acc_device_) {
          NNADAPTER_LOG(WARNING)
              << available_device_name
              << " is removed from the selected device list because "
                 "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE=TRUE";
        } else {
          NNADAPTER_LOG(WARNING) << "Add " << available_device_name;
          selected_devices_.push_back(available_device_info.first);
        }
      }
    }
  } else {
    NNADAPTER_LOG(WARNING) << "The property "
                              "'ANDROID_NNAPI_SELECTED_DEVICE_NAMES', "
                              "'ANDROID_NNAPI_SELECTED_DEVICE_IDS' and "
                              "'ANDROID_NNAPI_DISABLE_REFERENCE_DEVICE' are "
                              "only supported since Android sdk version "
                           << ANEURALNETWORKS_FEATURE_LEVEL_3
                           << " but the runtime's is "
                           << nnapi()->android_sdk_version;
  }
  // Relax computation float32 to float16
  if (nnapi()->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // ANDROID_NNAPI_RELAX_FP32_TO_FP16
    if (key_values.count(ANDROID_NNAPI_RELAX_FP32_TO_FP16)) {
      relax_fp32_to_fp16_ =
          string_parse<bool>(key_values[ANDROID_NNAPI_RELAX_FP32_TO_FP16]);
    } else {
      relax_fp32_to_fp16_ =
          GetBoolFromEnv(ANDROID_NNAPI_RELAX_FP32_TO_FP16, true);
    }
    NNADAPTER_LOG(INFO) << "relax_fp32_to_fp16: " << relax_fp32_to_fp16_;
  } else {
    NNADAPTER_LOG(WARNING) << "The property 'ANDROID_NNAPI_RELAX_FP32_TO_FP16' "
                              "is only supported since Android sdk version "
                           << ANEURALNETWORKS_FEATURE_LEVEL_2
                           << " but the runtime's is "
                           << nnapi()->android_sdk_version;
  }
}

Context::~Context() {}

Program::~Program() { Clear(); }

void Program::Clear() {
  if (execution_) {
    nnapi()->ANeuralNetworksExecution_free(execution_);
    execution_ = nullptr;
  }
  if (compilation_) {
    nnapi()->ANeuralNetworksCompilation_free(compilation_);
    compilation_ = nullptr;
  }
  if (model_) {
    nnapi()->ANeuralNetworksModel_free(model_);
    model_ = nullptr;
  }
  operand_indexes_.clear();
  operand_buffers_.clear();
  input_types_.clear();
  output_types_.clear();
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
    ConvertQuantizationSymmToAsymm(model);
    RestrictInputOutputQuantParams(model);
    ConvertDataLayoutNCHWToNHWC(model);
    ResolveOperationLiminations(model);
    NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  }
  // Convert the NNAdapter model to NNAPI model
  int result = nnapi()->ANeuralNetworksModel_create(&model_);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    NNADAPTER_LOG(FATAL) << "Failed to create a NNAPI Model(" << result << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  Converter converter(model_, &operand_indexes_);
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<uint32_t> input_operand_indexes;
  if (input_count > 0) {
    input_operand_indexes.resize(input_count);
    input_types_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      NNADAPTER_CHECK(operand_indexes_.find(operand) != operand_indexes_.end())
          << "No NNAPI operand found for input operand @0x" << std::hex
          << reinterpret_cast<int64_t>(operand);
      auto index = operand_indexes_[operand].front();
      NNADAPTER_CHECK_NE(index, INVALID_INDEX);
      NNADAPTER_VLOG(5) << "Found a NNAPI operand " << index
                        << " for input operand @0x" << std::hex
                        << reinterpret_cast<int64_t>(operand);
      input_operand_indexes[i] = index;
      input_types_[i] = operand->type;
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  std::vector<uint32_t> output_operand_indexes(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    NNADAPTER_CHECK(operand_indexes_.find(operand) != operand_indexes_.end())
        << "No NNAPI operand found for output operand @0x" << std::hex
        << reinterpret_cast<int64_t>(operand);
    auto index = operand_indexes_[operand].back();
    NNADAPTER_CHECK_NE(index, INVALID_INDEX);
    NNADAPTER_VLOG(5) << "Found a NNAPI operand " << index
                      << " for output operand @0x" << std::hex
                      << reinterpret_cast<int64_t>(operand);
    output_operand_indexes[i] = index;
    output_types_[i] = operand->type;
  }
  result = nnapi()->ANeuralNetworksModel_identifyInputsAndOutputs(
      model_,
      input_operand_indexes.size(),
      &input_operand_indexes[0],
      output_operand_indexes.size(),
      &output_operand_indexes[0]);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    nnapi()->ANeuralNetworksModel_free(model_);
    NNADAPTER_LOG(FATAL) << "Failed to identify the inputs and outputs("
                         << result << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  if (nnapi()->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_2) {
    result = nnapi()->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
        model_, context_->relax_fp32_to_fp16());
    if (result != ANEURALNETWORKS_NO_ERROR) {
      NNADAPTER_LOG(WARNING)
          << "Failed to relax computation float32 to float16!";
    }
  }
  result = nnapi()->ANeuralNetworksModel_finish(model_);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    nnapi()->ANeuralNetworksModel_free(model_);
    NNADAPTER_LOG(FATAL) << "Failed to finish the NNAPI model(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Build model
  if (!context_->selected_devices()->empty()) {
    result = nnapi()->ANeuralNetworksCompilation_createForDevices(
        model_,
        context_->selected_devices()->data(),
        context_->selected_devices()->size(),
        &compilation_);
  } else {
    result = nnapi()->ANeuralNetworksCompilation_create(model_, &compilation_);
  }
  if (result != ANEURALNETWORKS_NO_ERROR) {
    nnapi()->ANeuralNetworksModel_free(model_);
    NNADAPTER_LOG(FATAL) << "Failed to create a NNAPI Compilation(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  if (cache->token && cache->dir) {
    if (nnapi()->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_3) {
      result = nnapi()->ANeuralNetworksCompilation_setCaching(
          compilation_,
          cache->dir,
          reinterpret_cast<const uint8_t*>(cache->token));
      if (result != ANEURALNETWORKS_NO_ERROR) {
        NNADAPTER_LOG(WARNING) << "Failed to set the caching directory("
                               << cache->dir << ") and the model token("
                               << cache->token << ") to a NNAPI Compilation("
                               << result << ")!";
      }
    } else {
      NNADAPTER_LOG(WARNING) << "The compilation caching is only supported "
                                "since Android sdk version "
                             << ANEURALNETWORKS_FEATURE_LEVEL_3
                             << " but the runtime's is "
                             << nnapi()->android_sdk_version;
    }
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
  result = nnapi()->ANeuralNetworksCompilation_finish(compilation_);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    nnapi()->ANeuralNetworksModel_free(model_);
    nnapi()->ANeuralNetworksCompilation_free(compilation_);
    NNADAPTER_LOG(FATAL) << "Failed to compile the NNAPI Model(" << result
                         << ")!";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  }
  // Release the restored core::Model
  if (model_from_cache) {
    nnadapter::ClearModel(model);
    delete model;
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
  bool should_reset_execution = false;
  // Must reset execution before Android API 31
  if (!execution_ ||
      nnapi()->android_sdk_version <= ANEURALNETWORKS_FEATURE_LEVEL_4) {
    should_reset_execution = true;
  }
  if (should_reset_execution) {
    if (execution_) {
      nnapi()->ANeuralNetworksExecution_free(execution_);
      execution_ = nullptr;
    }
    ANeuralNetworksExecution* execution = nullptr;
    NNADAPTER_CHECK_EQ(
        nnapi()->ANeuralNetworksExecution_create(compilation_, &execution),
        ANEURALNETWORKS_NO_ERROR);
    if (nnapi()->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_5) {
      NNADAPTER_CHECK_EQ(nnapi()->ANeuralNetworksExecution_setReusable(
                             execution, /*reusable=*/true),
                         ANEURALNETWORKS_NO_ERROR);
    }
    execution_ = execution;
  }
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
    auto length = GetOperandTypeBufferLength(type);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(buffer));
    }
    if (should_reset_execution) {
      NNADAPTER_CHECK_EQ(nnapi()->ANeuralNetworksExecution_setInput(
                             execution_, arg.index, NULL, buffer, length),
                         ANEURALNETWORKS_NO_ERROR);
    }
  }
  std::vector<std::pair<void*, size_t>> output_buffers(output_count);
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
    auto length = GetOperandTypeBufferLength(*type);
    if (should_reset_execution) {
      NNADAPTER_CHECK_EQ(nnapi()->ANeuralNetworksExecution_setOutput(
                             execution_, arg.index, NULL, buffer, length),
                         ANEURALNETWORKS_NO_ERROR);
    }
    output_buffers[arg.index].first = buffer;
    output_buffers[arg.index].second = length;
  }
  auto start_time = GetCurrentUS();
  if (nnapi()->android_sdk_version < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    ANeuralNetworksEvent* event = nullptr;
    NNADAPTER_CHECK_EQ(
        nnapi()->ANeuralNetworksExecution_startCompute(execution_, &event),
        ANEURALNETWORKS_NO_ERROR);
    NNADAPTER_CHECK_EQ(nnapi()->ANeuralNetworksEvent_wait(event),
                       ANEURALNETWORKS_NO_ERROR);
    nnapi()->ANeuralNetworksEvent_free(event);
  } else {
    NNADAPTER_CHECK_EQ(nnapi()->ANeuralNetworksExecution_compute(execution_),
                       ANEURALNETWORKS_NO_ERROR);
  }
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  for (uint32_t i = 0; i < output_count; i++) {
    auto type = &output_types_[i];
    auto buffer = output_buffers[i].first;
    auto length = output_buffers[i].second;
    if (IsUInt8AsymmPerLayerQuantType(type->precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type->asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
