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

#include <string>
#include <vector>

#include "framework/program/op_desc.h"

namespace paddle_mobile {
namespace framework {

OpDesc::OpDesc(PaddleMobile__Framework__Proto__OpDesc *desc) {
  this->type_ = std::string(desc->type);
  for (int i = 0; i < desc->n_inputs; ++i) {
    PaddleMobile__Framework__Proto__OpDesc__Var *var = desc->inputs[i];
    std::vector<std::string> &args = inputs_[std::string(var->parameter)];
    for (int j = 0; j < var->n_arguments; ++j) {
      args.emplace_back(std::string(var->arguments[j]));
    }
  }

  for (int i = 0; i < desc->n_outputs; ++i) {
    PaddleMobile__Framework__Proto__OpDesc__Var *var = desc->outputs[i];
    std::vector<std::string> &args = outputs_[std::string(var->parameter)];
    for (int j = 0; j < var->n_arguments; ++j) {
      args.emplace_back(std::string(var->arguments[j]));
    }
  }

  for (int k = 0; k < desc->n_attrs; ++k) {
    PaddleMobile__Framework__Proto__OpDesc__Attr *attr = desc->attrs[k];
    std::string attr_name(attr->name);
    if (attr->type != PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__BLOCK) {
      attrs_[attr_name] = Attribute::GetAttrValue(attr);
    }
  }
}

const std::vector<std::string> &OpDesc::Input(const std::string &name) const {
  return inputs_.find(name)->second;
}

const std::vector<std::string> &OpDesc::Output(const std::string &name) const {
  return outputs_.find(name)->second;
}

Attribute OpDesc::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  return it->second;
}

std::unordered_map<std::string, Attribute> &OpDesc::GetAttrMap() {
  return attrs_;
}

Print &operator<<(Print &printer, const OpDesc &op_desc) {
  OpDesc &no_const_op_desc = const_cast<OpDesc &>(op_desc);
  printer << "inputs: \n";
  for (const auto &input : no_const_op_desc.GetInputs()) {
    printer << input.first << " : " << input.second << "\n";
  }

  printer << "outputs: \n";
  for (const auto &output : no_const_op_desc.GetOutputs()) {
    printer << output.first << " : " << output.second << "\n";
  }

  printer << "outputs: \n";
  for (const auto &attr : no_const_op_desc.GetAttrMap()) {
    printer << attr.first << " : " << attr.second << "\n";
  }
  return printer;
}

}  // namespace framework
}  // namespace paddle_mobile
