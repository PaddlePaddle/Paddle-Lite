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

#include "framework/selected_rows.h"

namespace paddle_mobile {
namespace framework {

struct ReAllocateVisitor {
  ReAllocateVisitor(framework::Tensor* tensor, const framework::DDim& dims)
      : tensor_(tensor), dims_(dims) {}

  template <typename T>
  void operator()() const {
    framework::Tensor cpu_tensor;
    T* ptr = cpu_tensor.mutable_data<T>(dims_);
    const T* old_ptr =
        tensor_->memory_size() == 0 ? nullptr : tensor_->data<T>();
    if (old_ptr != nullptr) {
      std::copy(old_ptr, old_ptr + tensor_->numel(), ptr);
    }
    tensor_->ShareDataWith(cpu_tensor);
  }

  framework::Tensor* tensor_;
  framework::DDim dims_;
};
// TensorCopyVisitor(value, i * value_width, *value_.get(),
//    index * value_width, value_width));
struct TensorCopyVisitor {
  TensorCopyVisitor(framework::Tensor* dst, int64_t dst_offset,
                    const framework::Tensor src, int64_t src_offset,
                    int64_t size)
      : dst_(dst),
        dst_offset_(dst_offset),
        src_(src),
        src_offset_(src_offset),
        size_(size) {}

  template <typename T>
  void operator()() const {
    // TODO(Yancey1989): support other place
    memory::Copy(dst_->mutable_data<T>() + dst_offset_,
                 src_.data<T>() + src_offset_, size_ * sizeof(T));
  }

  framework::Tensor* dst_;
  int64_t dst_offset_;
  framework::Tensor src_;
  int64_t src_offset_;
  int64_t size_;
};

bool SelectedRows::HasKey(int64_t key) const {
  return std::find(rows_.begin(), rows_.end(), key) == rows_.end() ? false
                                                                   : true;
}

// std::vector<int64_t> SelectedRows::Get(std::vector<int64_t> keys,
//                                       framework::Tensor* value) const {
//  PADDLE_MOBILE_ENFORCE(value->IsInitialized(),
//                 "The value tensor should be initialized.");
//  std::vector<int64_t> non_keys;
//  int64_t value_width = value_->numel() / value_->dims()[0];
//  PADDLE_MOBILE_ENFORCE(value_width == value->numel() / value->dims()[0],
//                    "output tensor should have the same shape with table "
//                    "execpt the dims[0].");
//
//  for (size_t i = 0; i < keys.size(); ++i) {
//    int64_t index = Index(keys[i]);
//    if (index == -1) {
//      non_keys.push_back(keys[i]);
//    } else {
//      framework::VisitDataType(
//          framework::ToDataType(value_->type()),
//          TensorCopyVisitor(value, i * value_width, *value_.get(),
//                            index * value_width, value_width));
//    }
//  }
//  return non_keys;
//}

// bool SelectedRows::Set(int64_t key, const framework::Tensor& value) {
//  PADDLE_MOBILE_ENFORCE(value.IsInitialized(), "The value should be
//  initialized."); if (value_->IsInitialized()) {
//    PADDLE_MOBILE_ENFORCE(
//        value.type() == value_->type(),
//        "The type of the value should be same with the original value");
//  }
//  PADDLE_MOBILE_ENFORCE(value.dims()[0] == static_cast<size_t>(1),
//                    "The first dim of value should be 1.");
//  auto index = Index(key);
//  bool is_new_key = false;
//  if (index == -1) {
//    rows_.push_back(key);
//    index = rows_.size() - 1;
//    is_new_key = true;
//    // whether need to resize the table
//    if (static_cast<int64_t>(rows_.size()) > value_->dims()[0]) {
//      auto dims = value_->dims();
//      dims[0] = (dims[0] + 1) << 1;
//      framework::VisitDataType(framework::ToDataType(value.type()),
//                               ReAllocateVisitor(value_.get(), dims));
//    }
//  }
//
//  framework::VisitDataType(
//      framework::ToDataType(value.type()),
//      TensorCopyVisitor(value_.get(),
//                        index * value_->numel() / value_->dims()[0], value,
//                        static_cast<int64_t>(0), value.numel()));
//  return is_new_key;
//}

}  // namespace framework
}  // namespace paddle_mobile
