/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

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
