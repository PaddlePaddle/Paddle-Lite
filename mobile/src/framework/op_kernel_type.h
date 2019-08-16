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

#include "framework/data_layout.h"
#include "framework/program/tensor_desc.h"

namespace paddle_mobile {
namespace framework {
struct OpKernelType {
  struct Hash {
    size_t operator()(const OpKernelType &key) const {
      int data_type = static_cast<int>(key.data_type_) << LEFT_SHIFT;
      int data_layout = static_cast<int>(key.data_layout_) << (LEFT_SHIFT * 2);

      std::hash<int> hasher;
      return hasher(data_type + data_layout);
    }
  };

  // place, data_type, library_type kinds less than 2^8
  constexpr static int LEFT_SHIFT = 8;

  VarType_Type data_type_;
  DataLayout data_layout_;

  OpKernelType(VarType_Type data_type,
               DataLayout data_layout = DataLayout::kAnyLayout)
      : data_type_(data_type), data_layout_(data_layout) {}

  bool operator==(const OpKernelType &o) const {
    return data_type_ == o.data_type_ && data_layout_ == o.data_layout_;
  }

  bool operator!=(const OpKernelType &o) const { return !(*this == o); }
};

inline bool NeedTransformLayout(const DataLayout &l, const DataLayout &r) {
  return l != DataLayout::kAnyLayout && r != DataLayout::kAnyLayout && l != r;
}

inline bool TransFromNeeded(const OpKernelType &l, const OpKernelType &r) {
  return (l.data_type_ != r.data_type_) ||
         NeedTransformLayout(l.data_layout_, r.data_layout_);
}

}  // namespace framework
}  // namespace paddle_mobile
