// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "lite/model_parser/base/op_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class OpDesc : public OpDescReadAPI {
 public:
  explicit OpDesc(proto::OpDesc* desc) : desc_(desc) { CHECK(desc_); }

  std::string Type() const override { return desc_->type()->str(); }

  // Get the arguments of parameter called `param`
  std::vector<std::string> Input(const std::string& param) const override {
    const auto& var = desc_->inputs()->LookupByKey(param.c_str());
    std::vector<std::string> args_vec;
    if (var->arguments()) {
      args_vec.reserve(var->arguments()->size());
      for (const auto& in : *var->arguments()) {
        args_vec.push_back(in->str());
      }
    }
    return args_vec;
  }

  std::vector<std::string> InputArgumentNames() const override {
    const auto& vars = desc_->inputs();
    std::vector<std::string> input_names_vec;
    if (vars) {
      input_names_vec.reserve(vars->size());
      for (const auto& in : *vars) {
        input_names_vec.push_back(in->parameter()->str());
      }
    }
    return input_names_vec;
  }

  std::vector<std::string> Output(const std::string& param) const override {
    const auto& var = desc_->outputs()->LookupByKey(param.c_str());
    std::vector<std::string> args_vec;
    if (var->arguments()) {
      args_vec.reserve(var->arguments()->size());
      for (const auto& out : *var->arguments()) {
        args_vec.push_back(out->str());
      }
    }
    return args_vec;
  }

  std::vector<std::string> OutputArgumentNames() const override {
    const auto& vars = desc_->outputs();
    std::vector<std::string> output_names_vec;
    if (vars) {
      output_names_vec.reserve(vars->size());
      for (const auto& out : *vars) {
        output_names_vec.push_back(out->parameter()->str());
      }
    }
    return output_names_vec;
  }

  bool HasAttr(const std::string& name) const override {
    return desc_->attrs()->LookupByKey(name.c_str()) == nullptr;
  }

  size_t AttrsSize() const { return desc_->attrs()->size(); }

  std::string AttrName(size_t idx) const {
    return desc_->attrs()->Get(idx)->name()->str();
  }

  OpDescAPI::AttrType GetAttrType(const std::string& name) const override {
    const auto& attr = desc_->attrs()->LookupByKey(name.c_str());
    CHECK(attr);
    return static_cast<OpDescAPI::AttrType>(attr->type());
  }

  OpDescAPI::AttrType GetAttrType(size_t idx) const {
    const auto& attr = desc_->attrs()->Get(idx);
    CHECK(attr);
    return static_cast<OpDescAPI::AttrType>(attr->type());
  }

  std::vector<std::string> AttrNames() const override {
    const auto& attrs = desc_->attrs();
    std::vector<std::string> attr_names_vec;
    if (attrs) {
      attr_names_vec.reserve(attrs->size());
      for (const auto& attr : *attrs) {
        attr_names_vec.push_back(attr->name()->str());
      }
    }
    return attr_names_vec;
  }

  template <typename T>
  T GetAttr(const std::string& name) const;

  template <typename T>
  T GetAttr(size_t idx) const;

  OpDesc() = delete;

 private:
  proto::OpDesc* desc_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
