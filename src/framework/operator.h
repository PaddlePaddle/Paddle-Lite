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

#include "attribute.h"
#include "block_desc.h"
#include "common/type_define.h"
#include "common/types.h"
#include "common/variant.h"
#include "op_info.h"
#include "op_kernel_type.h"
#include "paddle_mobile_object.h"
#include "scope.h"
#include "tensor.h"
#include "variable.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype> class OperatorBase : PaddleMobileObject {
public:
  OperatorBase(const std::string &type, const VariableNameMap &inputs,
               const VariableNameMap &outputs, const AttributeMap &attrs,
               std::shared_ptr<Scope> scope);
  virtual ~OperatorBase() {}
  virtual void Run();
  const VariableNameMap &Inputs() const { return inputs_; }
  const VariableNameMap &Outputs() const { return outputs_; }
  const std::string &Type() const { return type_; }
  const AttributeMap &Attrs() const { return attrs_; }

protected:
  std::shared_ptr<Scope> scope_;
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;

private:
  void CheckAllInputOutputSet() const;
  virtual void RunImpl() const = 0;
};

template <typename Dtype>
class OperatorWithKernel : public OperatorBase<Dtype> {
public:
  OperatorWithKernel(const std::string &type, const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     std::shared_ptr<Scope> scope)
      : OperatorBase<Dtype>(type, inputs, outputs, attrs, scope) {}
  virtual void InferShape() const = 0;

protected:
  virtual void RunImpl() const = 0;

private:
};

template <typename Dtype, typename P> class OpKernelBase : PaddleMobileObject {
public:
  virtual void Compute(const P &para) const = 0;

  virtual ~OpKernelBase() = default;
};

} // namespace framework
} // namespace paddle_mobile
