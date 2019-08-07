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

#ifdef REDUCE_PROD_OP

#pragma once

#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype>
class ReduceProdParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ReduceProdParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                  const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    input_ = OpParam::InputXFrom<GType>(inputs, *scope);
    output_ = OpParam::OutFrom<GType>(outputs, *scope);
    reduce_all_ = GetAttr<bool>("reduce_all", attrs);
    keep_dim_ = GetAttr<bool>("keep_dim", attrs);
    dim_ = GetAttr<std::vector<int>>("dim", attrs);
  }

  const GType *Input() const { return input_; }

  GType *Output() const { return output_; }

  bool isReduceAll() const { return reduce_all_; }

  bool isKeepDim() const { return keep_dim_; }

  const vector<int> getDim() const { return dim_; }

 private:
  GType *input_;
  GType *output_;
  bool reduce_all_;
  bool keep_dim_;
  std::vector<int> dim_;
};

DECLARE_KERNEL(ReduceProd, ReduceProdParam)

}  // namespace operators
}  // namespace paddle_mobile

#endif  // REDUCE_PROD_OP
