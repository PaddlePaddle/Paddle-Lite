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

//
// Created by liuRuiLong on 2018/5/4.
//
#include "framework/op_desc.h"
#include <string>
#include <vector>

namespace paddle_mobile {
namespace framework {

OpDesc::OpDesc(const proto::OpDesc &desc) : desc_(desc) {
  for (int i = 0; i < desc_.inputs_size(); ++i) {
    const proto::OpDesc::Var &var = desc_.inputs(i);
    std::vector<std::string> &args = inputs_[var.parameter()];
    int arg_size = var.arguments_size();
    for (int j = 0; j < arg_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }

  for (int i = 0; i < desc_.outputs_size(); ++i) {
    const proto::OpDesc::Var &var = desc_.outputs(i);
    std::vector<std::string> &args = outputs_[var.parameter()];
    int arg_size = var.arguments_size();
    for (int j = 0; j < arg_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }

  for (const proto::OpDesc::Attr &attr : desc_.attrs()) {
    std::string attr_name = attr.name();
    if (attr.type() != proto::AttrType::BLOCK) {
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

const std::unordered_map<std::string, Attribute> &OpDesc::GetAttrMap() const {
  return attrs_;
}

}  // namespace framework
}  // namespace paddle_mobile
