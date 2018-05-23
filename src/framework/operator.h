/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "common/type_define.h"
#include "common/types.h"
#include "common/variant.h"
#include "framework/attribute.h"
#include "framework/block_desc.h"
#include "framework/op_info.h"
#include "framework/op_kernel_type.h"
#include "framework/paddle_mobile_object.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"

namespace paddle_mobile {
namespace framework {
static std::unordered_map<
    std::string, std::pair<std::vector<std::string>, std::vector<std::string>>>
    op_input_output_key = {{"conv2d", {{"Input"}, {"Output"}}},
                           {"relu", {{"X"}, {"Out"}}},
                           {"softmax", {{"X"}, {"Out"}}},
                           {"mul", {{"X"}, {"Out"}}},
                           {"elementwise_add", {{"X", "Y"}, {"Out"}}},
                           {"pool2d", {{"X"}, {"Out"}}},
                           {"batch_norm", {{"X"}, {"Y"}}},
                           {"lrn", {{"X"}, {"Out"}}},
                           {"concat", {{"X"}, {"Out"}}},
                           {"feed", {{"X"}, {"Out"}}},
                           {"fetch", {{"X"}, {"Out"}}}};

template <typename Dtype>
class OperatorBase : PaddleMobileObject {
 public:
  OperatorBase(const std::string &type, const VariableNameMap &inputs,
               const VariableNameMap &outputs, const AttributeMap &attrs,
               std::shared_ptr<Scope> scope);
  virtual ~OperatorBase() {}
  virtual void Run() const = 0;

  const VariableNameMap &Inputs() const { return inputs_; }
  const VariableNameMap &Outputs() const { return outputs_; }
  const std::string &Type() const { return type_; }
  const AttributeMap &Attrs() const { return attrs_; }
  void ClearVariables(const std::vector<std::string> &var_names) const {
    if (this->scope_) {
      this->scope_->EraseVars(var_names);
    }
  }

 protected:
  std::shared_ptr<Scope> scope_;
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;

 private:
  void CheckAllInputOutputSet() const;
};

template <typename Dtype>
class OperatorWithKernel : public OperatorBase<Dtype> {
 public:
  OperatorWithKernel(const std::string &type, const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     std::shared_ptr<Scope> scope)
      : OperatorBase<Dtype>(type, inputs, outputs, attrs, scope) {}
  virtual void InferShape() const = 0;
  virtual void Run() const = 0;
};

template <typename Dtype, typename P>
class OpKernelBase : PaddleMobileObject {
 public:
  virtual void Compute(const P &para) const = 0;

  virtual ~OpKernelBase() = default;
};

#define DEFINE_OP_CONSTRUCTOR(cls, parent_cls)                                 \
  cls(const std::string &type, const ::paddle_mobile::VariableNameMap &inputs, \
      const ::paddle_mobile::VariableNameMap &outputs,                         \
      const ::paddle_mobile::framework::AttributeMap &attrs,                   \
      std::shared_ptr<::paddle_mobile::framework::Scope> scope)                \
      : parent_cls<Dtype, T>(type, inputs, outputs, attrs, scope) {}

}  // namespace framework
}  // namespace paddle_mobile
