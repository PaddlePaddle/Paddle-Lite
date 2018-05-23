/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
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
      //      if (attr.type() == proto::AttrType::INT){
      //        std::cout << " attrName " << attr_name << " " <<
      //        attrs_[attr_name].Get<int>() << std::endl;
      //      }
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
