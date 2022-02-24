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

#pragma once

#include <limits.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/huawei_ascend_npu/utility.h"
#include "op_proto/built-in/inc/all_ops.h"
#include "utility/debug.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_ascend_npu {

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

class Converter {
 public:
  explicit Converter(
      std::map<core::Operand*, std::vector<std::shared_ptr<Operator>>>*
          operators)
      : operators_(operators) {}
  ~Converter() {}

  // Convert a NNAdapter model to GE graph and operators
  int Apply(core::Model* model);
  // Mapping a GE operator to a NNAdapter operand
  std::shared_ptr<Operator> GetMappedOperator(core::Operand* operand);
  std::shared_ptr<Operator> UpdateOperatorMap(core::Operand* operand,
                                              std::shared_ptr<Operator> op);
  template <typename T>
  std::shared_ptr<T> AddOperator(core::Operand* operand = nullptr,
                                 const std::string& custom_name = "") {
    std::string operand_id = OperandIdToString(operand);
    std::string operator_name = string_format("op_%d_%s_%s_%s",
                                              operator_index_++,
                                              typeid(T).name(),
                                              operand_id.c_str(),
                                              custom_name.c_str());
    return std::make_shared<T>(operator_name);
  }
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
  std::shared_ptr<Operator> AddUInt64ConstantOperator(
      const uint64_t* values, const std::vector<int32_t>& dimensions);
  std::shared_ptr<Operator> AddUInt64ConstantOperator(
      const std::vector<uint64_t>& values,
      const std::vector<int32_t>& dimensions = {});
  std::shared_ptr<Operator> AddZeroConstantOperator(
      NNAdapterOperandPrecisionCode precision,
      const std::vector<int32_t>& dimensions);
  // Convert a constant and model input operand and map to a operator
  std::shared_ptr<Operator> ConvertOperand(
      core::Operand* operand, std::vector<int32_t> dimensions = {});

 private:
  std::map<core::Operand*, std::vector<std::shared_ptr<Operator>>>* operators_{
      nullptr};
  // Only for generating the unique name for GE operator
  uint32_t operator_index_{0};
};

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
    converter->UpdateOperatorMap(                                              \
        dst, std::make_shared<Operator>(src, tensor_desc, #name, -1));         \
  })

#define MAP_DYNAMIC_OUTPUT(src, name, index, dst)                              \
  ({                                                                           \
    auto shape = ge::Shape();                                                  \
    auto format = ge::FORMAT_NCHW;                                             \
    auto dtype = ConvertToGEPrecision(dst->type.precision);                    \
    auto tensor_desc = std::make_shared<ge::TensorDesc>(shape, format, dtype); \
    src->update_dynamic_output_desc_##name(index, *tensor_desc);               \
    converter->UpdateOperatorMap(                                              \
        dst, std::make_shared<Operator>(src, tensor_desc, #name, index));      \
  })

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
