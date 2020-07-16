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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lite/model_parser/base/op_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/model_parser/flatbuffers/vector_view.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class OpDesc : public OpDescAPI {
 public:
  explicit OpDesc(proto::OpDesc const* desc) : desc_(desc) { CHECK(desc_); }

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
    return desc_->attrs()->LookupByKey(name.c_str()) != nullptr;
  }

  size_t AttrsSize() const { return desc_->attrs()->size(); }

  std::string AttrName(size_t idx) const {
    return desc_->attrs()->Get(idx)->name()->str();
  }

  OpDescAPI::AttrType GetAttrType(const std::string& name) const override {
    const auto& attr = desc_->attrs()->LookupByKey(name.c_str());
    CHECK(attr) << "Can not find attr: " << name;
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
  typename lite::OpDataTypeTrait<T, Flatbuffers>::RT GetAttr(
      const std::string& name) const;

  template <typename T>
  typename lite::OpDataTypeTrait<T, Flatbuffers>::RT GetAttr(size_t idx) const;

 private:
  proto::OpDesc const* desc_;

  // To reduce overhead, we expect to use namespace aliasing to make cpp::Desc
  // and flatbuffers::Desc replace each other. However, there is no direct
  // inheritance relationship between the two data types, and the read-only
  // version of flatbuffers lacks some write implementations. Therefore, at
  // present, we are temporarily providing a default interface that triggers
  // execution-time errors to avoid type ambiguity and compile-time errors
  // caused by different building options.

 public:
  OpDesc() { NotImplemented(); }
  bool HasInput(const std::string& param) const {
    return desc_->inputs()->LookupByKey(param.c_str()) != nullptr;
  }

  const std::map<std::string, std::vector<std::string>>& inputs() const {
    NotImplemented();
    return inputs_;
  }
  const std::map<std::string, std::vector<std::string>>& outputs() const {
    NotImplemented();
    return outputs_;
  }
  std::map<std::string, std::vector<std::string>>* mutable_inputs() {
    NotImplemented();
    return &inputs_;
  }
  std::map<std::string, std::vector<std::string>>* mutable_outputs() {
    NotImplemented();
    return &outputs_;
  }

  std::vector<std::string> input_vars() const {
    NotImplemented();
    return std::vector<std::string>();
  }

  std::vector<std::string> output_vars() const {
    NotImplemented();
    return std::vector<std::string>();
  }

  bool HasOutput(const std::string& param) const {
    NotImplemented();
    return false;
  }

  const std::map<std::string, Any>& attrs() const {
    NotImplemented();
    return attrs_;
  }
  const std::map<std::string, AttrType>& attr_types() const {
    NotImplemented();
    return attr_types_;
  }

 private:
  void NotImplemented() const {
    LOG(FATAL) << "The additional interfaces of OpDesc is temporarily "
                  "unavailable in read-only mode.";
  }
  std::string type_;
  std::map<std::string, std::vector<std::string>> inputs_;
  std::map<std::string, std::vector<std::string>> outputs_;
  std::map<std::string, Any> attrs_;
  std::map<std::string, AttrType> attr_types_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
