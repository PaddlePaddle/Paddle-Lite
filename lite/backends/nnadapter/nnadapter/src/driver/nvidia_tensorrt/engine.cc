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
#include <algorithm>
#include <unordered_set>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

void Program::Clear() {
  sub_models_.clear();
  sub_programs_.clear();
  input_tensors_.clear();
  temporary_tensors_.clear();
  output_tensors_.clear();
  input_types_.clear();
  output_types_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (cache->buffer.empty()) {
    for (auto& operand : model->input_operands) {
      input_types_.push_back(operand->type);
    }
    for (auto& operand : model->output_operands) {
      output_types_.push_back(operand->type);
    }
  } else {
    input_types_ = cache->input_types;
    output_types_ = cache->output_types;
  }
  for (auto& type : input_types_) {
    ConvertDynamicDimensions(&type.back());
  }
  for (auto& type : output_types_) {
    ConvertDynamicDimensions(&type.back());
  }
  // Partion model
  std::vector<std::pair<int, std::unordered_set<core::Operation*>>>
      supported_operations{{0, {}}, {1, {}}};
  for (auto& operation : model->operations) {
    // // Set softmax to cuda device
    // if (operation.type == NNADAPTER_SOFTMAX) {
    //   supported_operations[1].second.insert(&operation);
    //   continue;
    // }
    supported_operations[0].second.insert(&operation);
  }
  PartitionModelIntoSubmodels(model, supported_operations, &sub_models_);
  NNADAPTER_CHECK(!sub_models_.empty());
  for (auto& sub_model : sub_models_) {
    if (sub_model.first == 0) {
      sub_programs_.emplace_back(
          new TensorrtProgram(context_, std::get<0>(sub_model.second), cache));
    } else {
      sub_programs_.emplace_back(
          new CudaProgram(context_, std::get<0>(sub_model.second), cache));
    }
    sub_programs_.back()->Build();
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

static void SetTensor(Tensor* tensor,
                      void* host_ptr,
                      const NNAdapterOperandType& type) {
  auto dims = type.dimensions;
  std::vector<int32_t> shape(dims.data, dims.data + dims.count);
  tensor->Resize(shape);
  tensor->SetDateType(ConvertToNVDataType(type.precision));
  uint32_t length =
      tensor->Length() * GetOperandPrecisionDataLength(type.precision);
  NNADAPTER_CHECK_EQ(
      cudaMemcpy(tensor->Data(), host_ptr, length, cudaMemcpyHostToDevice),
      cudaSuccess);
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  // 1. Feed inputs
  for (size_t i = 0; i < input_types_.size(); i++) {
    // Get input info
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    auto type = input_types_.at(i);
    auto host_ptr = arg->access(arg->memory, &type);
    NNADAPTER_CHECK(host_ptr);
    // Fill input tensor
    int index = static_cast<int>(i) - static_cast<int>(input_types_.size());
    if (!input_tensors_.count(index)) {
      input_tensors_[index] = std::shared_ptr<Tensor>(new Tensor());
    }
    SetTensor(input_tensors_[index].get(), host_ptr, type);
  }
  // 2. Execute sub_programs_ in order
  for (size_t i = 0; i < sub_programs_.size(); i++) {
    auto input_indexes = std::get<2>(sub_models_.at(i).second);
    std::sort(input_indexes.begin(), input_indexes.end());
    auto output_indexes = std::get<3>(sub_models_.at(i).second);
    std::sort(output_indexes.begin(), output_indexes.end());
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    std::vector<std::shared_ptr<Tensor>> output_tensors;
    // Find inputs
    for (auto input_index : input_indexes) {
      if (input_index < 0) {
        NNADAPTER_CHECK(input_tensors_.count(input_index));
        input_tensors.push_back(input_tensors_.at(input_index));
      } else {
        NNADAPTER_CHECK(temporary_tensors_.count(input_index));
        input_tensors.push_back(temporary_tensors_.at(input_index));
      }
    }
    // Init outputs
    for (auto output_index : output_indexes) {
      if (output_index < 0) {
        if (!output_tensors_.count(output_index)) {
          output_tensors_[output_index] = std::shared_ptr<Tensor>(new Tensor());
        }
        output_tensors.push_back(output_tensors_[output_index]);
      } else {
        if (!temporary_tensors_.count(output_index)) {
          temporary_tensors_[output_index] =
              std::shared_ptr<Tensor>(new Tensor());
        }
        output_tensors.push_back(temporary_tensors_[output_index]);
      }
    }
    sub_programs_[i]->Execute(&input_tensors, &output_tensors);
  }
  // 3. Fetch outputs
  for (size_t i = 0; i < output_types_.size(); i++) {
    auto arg = FindArgumentByIndex(output_arguments, i, output_count);
    NNADAPTER_CHECK(arg) << "Output argument " << i << " does not exist!";
    int index = static_cast<int>(i) - static_cast<int>(output_types_.size());
    auto dims = output_tensors_.at(index)->Dims();
    NNAdapterOperandType type = output_types_.at(i);
    type.dimensions.count = dims.size();
    memcpy(type.dimensions.data, dims.data(), dims.size() * sizeof(int32_t));
    auto host_ptr = arg->access(arg->memory, &type);
    auto length = GetOperandTypeBufferLength(type);
    NNADAPTER_CHECK_EQ(cudaMemcpy(host_ptr,
                                  output_tensors_.at(index)->Data(),
                                  length,
                                  cudaMemcpyDeviceToHost),
                       cudaSuccess);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
