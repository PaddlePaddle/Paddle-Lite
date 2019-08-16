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

#pragma once

#include <functional>
#include <string>
#include "common/log.h"
#include "common/type_define.h"
#include "framework/scope.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class OperatorBase;

template <typename Dtype>
using OpCreator = std::function<framework::OperatorBase<Dtype> *(
    const std::string & /*type*/, const VariableNameMap & /*inputs*/,
    const VariableNameMap & /*outputs*/,
    const framework::AttributeMap & /*attrs*/, framework::Scope * /*scope*/)>;

template <typename Dtype>
struct OpInfo {
  OpCreator<Dtype> creator_;
  const OpCreator<Dtype> &Creator() const {
    PADDLE_MOBILE_ENFORCE(creator_ != nullptr,
                          "Operator Creator has not been registered");
    return creator_;
  }
};

template <typename Dtype>
class OpInfoMap {
 public:
  static OpInfoMap<Dtype> *Instance() {
    static OpInfoMap<Dtype> *s_instance = nullptr;
    if (s_instance == nullptr) {
      s_instance = new OpInfoMap();
    }
    return s_instance;
  }

  bool Has(const std::string &op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string &type, const OpInfo<Dtype> &info) {
    PADDLE_MOBILE_ENFORCE(!Has(type), "Operator %s has been registered",
                          type.c_str());
    map_.insert({type, info});
  }

  const OpInfo<Dtype> &Get(const std::string &type) const {
    auto op_info_ptr = GetNullable(type);
    PADDLE_MOBILE_ENFORCE(op_info_ptr != nullptr,
                          "Operator %s has not been registered", type.c_str());
    return *op_info_ptr;
  }

  const OpInfo<Dtype> *GetNullable(const std::string &type) const {
    auto it = map_.find(type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  const std::unordered_map<std::string, OpInfo<Dtype>> &map() const {
    return map_;
  }

  std::unordered_map<std::string, OpInfo<Dtype>> *mutable_map() {
    return &map_;
  }

 private:
  OpInfoMap() = default;
  std::unordered_map<std::string, OpInfo<Dtype>> map_;
};

}  // namespace framework
}  // namespace paddle_mobile
