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

#include "common/type_define.h"
#include "framework.pb.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
struct OpInfo {
  OpCreator<Dtype> creator_;
  const OpCreator<Dtype>& Creator() const {
    //    PADDLE_ENFORCE_NOT_NULL(creator_,
    //                            "Operator Creator has not been registered");
    return creator_;
  }
};

template <typename Dtype>
class OpInfoMap;

template <typename Dtype>
static OpInfoMap<Dtype>* g_op_info_map = nullptr;

template <typename Dtype>
class OpInfoMap {
 public:
  static OpInfoMap& Instance() {
    if (g_op_info_map<Dtype> == nullptr) {
      g_op_info_map<Dtype> = new OpInfoMap();
    }
    return *g_op_info_map<Dtype>;
  };

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string& type, const OpInfo<Dtype>& info) {
    //    PADDLE_ENFORCE(!Has(type), "Operator %s has been registered", type);
    map_.insert({type, info});
  }

  const OpInfo<Dtype>& Get(const std::string& type) const {
    auto op_info_ptr = GetNullable(type);
    //    PADDLE_ENFORCE_NOT_NULL(op_info_ptr, "Operator %s has not been
    //    registered",
    //                            type);
    return *op_info_ptr;
  }

  const OpInfo<Dtype>* GetNullable(const std::string& type) const {
    auto it = map_.find(type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  const std::unordered_map<std::string, OpInfo<Dtype>>& map() const {
    return map_;
  }

  std::unordered_map<std::string, OpInfo<Dtype>>* mutable_map() {
    return &map_;
  }

 private:
  OpInfoMap() = default;
  std::unordered_map<std::string, OpInfo<Dtype>> map_;

  //  DISABLE_COPY_AND_ASSIGN(OpInfoMap);
};

}  // namespace framework
}  // namespace paddle_mobile