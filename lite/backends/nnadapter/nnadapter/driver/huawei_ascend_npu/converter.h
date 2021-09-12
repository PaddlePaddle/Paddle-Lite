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

#pragma once

#include <limits.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "driver/huawei_ascend_npu/utility.h"
#include "op_proto/built-in/inc/all_ops.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_ascend_npu {

class Device {
 public:
  Device();
  ~Device();
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  int GetFirstDeviceID() {
    return selected_device_ids_.empty() ? 0 : selected_device_ids_[0];
  }
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  std::vector<int> selected_device_ids_;
};

class Operator {
 public:
  explicit Operator(std::shared_ptr<ge::Operator> op,
                    std::shared_ptr<ge::TensorDesc> tensor_desc,
                    const std::string& component_name = "",
                    int component_index = -1)
      : op_(op),
        tensor_desc_(tensor_desc),
        component_name_(component_name),
        component_index_(component_index) {}
  std::shared_ptr<ge::Operator> op() { return op_; }
  std::shared_ptr<ge::TensorDesc> tensor_desc() { return tensor_desc_; }
  std::string component_name() { return component_name_; }
  int component_index() { return component_index_; }
  ~Operator() {}

 private:
  std::shared_ptr<ge::Operator> op_{nullptr};
  std::shared_ptr<ge::TensorDesc> tensor_desc_;
  std::string component_name_;
  int component_index_{-1};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(hal::Model* model, hal::Cache* cache);
  int Execute(uint32_t input_count,
              hal::Argument* input_arguments,
              uint32_t output_count,
              hal::Argument* output_arguments);

 private:
  void Clear();
  int32_t global_idx;
  // Operand converters

  std::string GetOperatorName(hal::Operand* operand);
  std::shared_ptr<Operator> GetMappedOperator(hal::Operand* operand);
  std::shared_ptr<Operator> UpdateOperatorMap(hal::Operand* operand,
                                              std::shared_ptr<Operator> op);
  std::shared_ptr<Operator> AddConstantOperator(
      const void* values,
      NNAdapterOperandPrecisionCode precision,
      const std::vector<int32_t>& dimensions = {});
  std::shared_ptr<Operator> AddInt32ConstantOperator(const int32_t values);
  std::shared_ptr<Operator> AddInt32ConstantOperator(
      const int32_t* values, const std::vector<int32_t>& dimensions);
  std::shared_ptr<Operator> AddInt32ConstantOperator(
      const std::vector<int32_t>& values,
      const std::vector<int32_t>& dimensions = {});
  std::shared_ptr<Operator> AddFloat32ConstantOperator(
      const float* values, const std::vector<int32_t>& dimensions);
  std::shared_ptr<Operator> AddFloat32ConstantOperator(
      const std::vector<float>& values,
      const std::vector<int32_t>& dimensions = {});
  // Convert a constant and model input operand and map to a operator
  std::shared_ptr<Operator> ConvertOperand(
      hal::Operand* operand, std::vector<int32_t> dimensions = {});

  // Operation converters
  int ConvertConv2D(hal::Operation* operation);
  int ConvertFullyConnected(hal::Operation* operation);
  int ConvertFill(hal::Operation* operation);
  int ConvertPool2D(hal::Operation* operation);
  int ConvertElementwise(hal::Operation* operation);
  int ConvertSoftmax(hal::Operation* operation);
  int ConvertCumSum(hal::Operation* operation);
  int ConvertActivation(hal::Operation* operation);
  int ConvertReshape(hal::Operation* operation);
  int ConvertTranspose(hal::Operation* operation);
  int ConvertConcat(hal::Operation* operation);
  int ConvertSplit(hal::Operation* operation);
  int ConvertPow(hal::Operation* operation);
  int ConvertBatchNormalization(hal::Operation* operation);
  int ConvertClip(hal::Operation* operation);
  int ConvertLeakyRelu(hal::Operation* operation);
  int ConvertSlice(hal::Operation* operation);
  int ConvertReduceMean(hal::Operation* operation);
  int ConvertExpand(hal::Operation* operation);
  int ConvertRange(hal::Operation* operation);
  int ConvertCast(hal::Operation* operation);
  int ConvertShape(hal::Operation* operation);
  int ConvertStack(hal::Operation* operation);
  int ConvertAssign(hal::Operation* operation);
  int ConvertResizeNearest(hal::Operation* operation);
  int ConvertResizeLinear(hal::Operation* operation);
  int ConvertInstanceNormalization(hal::Operation* operation);
  int ConvertLpNormalization(hal::Operation* operation);
  int ConvertDeformableConv2d(hal::Operation* operation);
  int ConvertHardSwish(hal::Operation* operation);
  int ConvertHardSigmoid(hal::Operation* operation);
  int ConvertSqueeze(hal::Operation* operation);
  int ConvertUnsqueeze(hal::Operation* operation);
  int ConvertPad(hal::Operation* operation);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to GE operator
  std::map<hal::Operand*, std::vector<std::shared_ptr<Operator>>> operators_;
  std::shared_ptr<AclModelClient> model_client_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
};

// Create operator
#define CREATE_OPERATOR(op_type, output_operand, custom_name)              \
  {                                                                        \
    std::string op_name;                                                   \
    auto operand_name = GetOperatorName(output_operand);                   \
    if (custom_name.empty()) {                                             \
      op_name = string_format("%s_%s", op_type, output_operand);           \
    } else {                                                               \
      op_name =                                                            \
          string_format("%s_%s_%s", op_type, custom_name, output_operand); \
    }                                                                      \
    std::make_shared<ge::op::##op_type>(op_name);                          \
  }

// Set one of dynamic inputs of a ge::Operator and update its tensor desc
#define SET_INPUT(dst, name, src)                                 \
  {                                                               \
    auto value = src->op();                                       \
    auto tensor_desc = src->tensor_desc();                        \
    auto cmpt_name = src->component_name();                       \
    auto cmpt_index = src->component_index();                     \
    if (cmpt_name.empty()) {                                      \
      dst->set_input_##name(*value);                              \
    } else {                                                      \
      if (cmpt_index >= 0) {                                      \
        cmpt_name += string_format("%d", cmpt_index);             \
      }                                                           \
      dst->set_input_##name##_by_name(*value, cmpt_name.c_str()); \
    }                                                             \
    dst->update_input_desc_##name(*tensor_desc);                  \
  }

// Map one of dynamic outputs to a operand and update its tensor desc
#define SET_DYNAMIC_INPUT(dst, name, index, src)                       \
  {                                                                    \
    auto value = src->op();                                            \
    auto tensor_desc = src->tensor_desc();                             \
    auto cmpt_name = src->component_name();                            \
    auto cmpt_index = src->component_index();                          \
    if (cmpt_name.empty()) {                                           \
      dst->set_dynamic_input_##name(index, *value);                    \
    } else {                                                           \
      if (cmpt_index >= 0) {                                           \
        cmpt_name += string_format("%d", cmpt_index);                  \
      }                                                                \
      dst->set_dynamic_input_##name(index, *value, cmpt_name.c_str()); \
    }                                                                  \
    dst->update_dynamic_input_desc_##name(index, *tensor_desc);        \
  }

#define MAP_OUTPUT(src, name, dst)                                             \
  ({                                                                           \
    auto shape = ge::Shape();                                                  \
    auto format = ge::FORMAT_NCHW;                                             \
    auto dtype = ConvertToGEPrecision(dst->type.precision);                    \
    auto tensor_desc = std::make_shared<ge::TensorDesc>(shape, format, dtype); \
    src->update_output_desc_##name(*tensor_desc);                              \
    UpdateOperatorMap(                                                         \
        dst, std::make_shared<Operator>(src, tensor_desc, #name, -1));         \
  })

#define MAP_DYNAMIC_OUTPUT(src, name, index, dst)                              \
  ({                                                                           \
    auto shape = ge::Shape();                                                  \
    auto format = ge::FORMAT_NCHW;                                             \
    auto dtype = ConvertToGEPrecision(dst->type.precision);                    \
    auto tensor_desc = std::make_shared<ge::TensorDesc>(shape, format, dtype); \
    src->update_dynamic_output_desc_##name(index, *tensor_desc);               \
    UpdateOperatorMap(                                                         \
        dst, std::make_shared<Operator>(src, tensor_desc, #name, index));      \
  })

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
