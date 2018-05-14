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

#include "data_layout.h"
#include "framework.pb.h"

namespace paddle_mobile {
namespace framework {
struct OpKernelType {
  struct Hash {
    size_t operator()(const OpKernelType& key) const {
      int data_type = static_cast<int>(key.data_type_) << LEFT_SHIFT;
      int data_layout = static_cast<int>(key.data_layout_) << (LEFT_SHIFT * 2);

      std::hash<int> hasher;
      return hasher(data_type + data_layout);
    }
  };

  // place, data_type, library_type kinds less than 2^8
  constexpr static int LEFT_SHIFT = 8;

  proto::VarType::Type data_type_;
  DataLayout data_layout_;

  OpKernelType(proto::VarType::Type data_type, DataLayout data_layout = DataLayout::kAnyLayout)
          : data_type_(data_type),
            data_layout_(data_layout) {}

  bool operator==(const OpKernelType& o) const {
    return data_type_ == o.data_type_ && data_layout_ == o.data_layout_;
  }

  bool operator!=(const OpKernelType& o) const { return !(*this == o); }
};

inline bool NeedTransformLayout(const DataLayout& l, const DataLayout& r) {
  return l != DataLayout::kAnyLayout && r != DataLayout::kAnyLayout && l != r;
}

inline bool TransFromNeeded(const OpKernelType& l, const OpKernelType& r) {
  return (l.data_type_ != r.data_type_) ||
         NeedTransformLayout(l.data_layout_, r.data_layout_);
}

} // framework
} // paddle_mobile
