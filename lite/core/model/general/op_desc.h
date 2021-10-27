// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <utility>
#include <vector>
#include "lite/core/model/base/apis.h"
#include "lite/utils/any.h"
#include "lite/utils/varient.h"

namespace paddle {
namespace lite {
namespace general {

/*
 * The general::OpDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::OpDesc.
 */
class OpDesc : public OpDescAPI {
 public:
  using attrs_t = std::map<std::string, Any>;
  using attr_types_t = std::map<std::string, AttrType>;

 protected:
  std::string type_;
  std::map<std::string, std::vector<std::string>> inputs_;
  std::map<std::string, std::vector<std::string>> outputs_;
  std::map<std::string, Any> attrs_;
  std::map<std::string, AttrType> attr_types_;

 public:
  OpDesc() = default;

  std::string Type() const override { return type_; }
  void SetType(const std::string& x) override { type_ = x; }

  const std::map<std::string, std::vector<std::string>>& inputs() const {
    return inputs_;
  }
  const std::map<std::string, std::vector<std::string>>& outputs() const {
    return outputs_;
  }
  std::map<std::string, std::vector<std::string>>* mutable_inputs() {
    return &inputs_;
  }
  std::map<std::string, std::vector<std::string>>* mutable_outputs() {
    return &outputs_;
  }

  bool HasInput(const std::string& param) const {
    auto it = inputs_.find(param);
    return it != inputs_.end();
  }

  std::vector<std::string> Input(const std::string& param) const override;

  std::vector<std::string> InputArgumentNames() const override;
  std::vector<std::string> OutputArgumentNames() const override;

  std::vector<std::string> input_vars() const;

  std::vector<std::string> output_vars() const;

  bool HasOutput(const std::string& param) const;

  std::vector<std::string> Output(const std::string& param) const override;

  void SetInput(const std::string& param,
                const std::vector<std::string>& args) override {
    inputs_[param] = args;
  }

  void SetOutput(const std::string& param,
                 const std::vector<std::string>& args) override {
    outputs_[param] = args;
  }

  bool HasAttr(const std::string& name) const override {
    return attrs_.count(name);
  }

  AttrType GetAttrType(const std::string& name) const override {
    auto it = attr_types_.find(name);
    CHECK(it != attr_types_.end());
    return it->second;
  }

  std::vector<std::string> AttrNames() const override {
    std::vector<std::string> res;
    for (const auto& x : attrs_) {
      res.push_back(x.first);
    }
    return res;
  }

  template <typename T>
  void SetAttr(const std::string& name, const T& v) {
    attr_types_[name] = OpDataTypeTrait<T>::AT;
    attrs_[name].set(v);
  }

  template <typename T>
  T GetAttr(const std::string& name) const {
    auto it = attrs().find(name);
    CHECK(it != attrs().end()) << "No attributes called " << name
                               << " found for " << Type();
    auto attr_it = attr_types().find(name);
    CHECK(attr_it != attr_types().end());
    auto pair = std::make_pair(it, attr_it);
    CHECK(pair.second->second == OpDataTypeTrait<T>::AT)
        << "required type is " << OpDataTypeTrait<T>::ATN
        << " not match the true type";
    return pair.first->second.get<T>();
  }

  void DeleteAttr(const std::string& name) {
    if (attrs_.count(name) > 0) {
      attrs_.erase(name);
    }
  }

  const std::map<std::string, Any>& attrs() const { return attrs_; }
  const std::map<std::string, AttrType>& attr_types() const {
    return attr_types_;
  }
};

}  // namespace general
}  // namespace lite
}  // namespace paddle
