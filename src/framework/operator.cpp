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

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
vector<string> OperatorBase<Dtype>::GetOutKeys() const {
  auto it = op_input_output_key.find(type_);
  if (it == op_input_output_key.end()) {
    DLOG << type_ << " has no outputs";
    return {};
  }
  return it->second.second;
}

template <typename Dtype>
vector<string> OperatorBase<Dtype>::GetInputKeys() const {
  auto it = op_input_output_key.find(type_);
  if (it == op_input_output_key.end()) {
    DLOG << type_ << " has no outputs";
    return {};
  }
  return it->second.first;
}

template <typename Dtype>
OperatorBase<Dtype>::OperatorBase(const std::string &type,
                                  const VariableNameMap &inputs,
                                  const VariableNameMap &outputs,
                                  const AttributeMap &attrs,
                                  std::shared_ptr<Scope> scope)
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      scope_(scope) {
  CheckAllInputOutputSet();
}

template <typename Dtype>
void OperatorBase<Dtype>::CheckAllInputOutputSet() const {}

template <typename Dtype>
void OperatorBase<Dtype>::Run() const {
  RunImpl();
#ifdef PADDLE_MOBILE_DEBUG
  vector<string> input_keys = GetInputKeys();
  for (const auto key : input_keys) {
    Tensor *input = GetVarValue<framework::LoDTensor>(key, inputs_, *scope_);
    if (input) DLOG << type_ << " input- " << key << "=" << *input;
  }
  vector<string> output_keys = GetOutKeys();
  for (const auto key : output_keys) {
    Tensor *out_ = GetVarValue<framework::LoDTensor>(key, outputs_, *scope_);
    DLOG << type_ << " output- " << key << "=" << *out_;
  }
#endif
}

template class OperatorBase<CPU>;
template class OperatorBase<FPGA>;
template class OperatorBase<GPU_MALI>;

}  // namespace framework
}  // namespace paddle_mobile
