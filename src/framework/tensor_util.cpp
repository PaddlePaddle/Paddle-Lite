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

#include "tensor_util.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace paddle_mobile {
namespace framework {

void TensorCopy(const Tensor &src, Tensor *dst) {
  //  VLOG(3) << "TensorCopy " << src.dims() << " from " <<
  //  src.place() << " to
  //  "
  //          << dst_place;
  src.check_memory_size();

  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_ptr = src.data<void>();

  auto dst_ptr = dst->mutable_data(src.type());

  auto size = src.numel() * SizeOfType(src.type());

  memory::Copy(dst_ptr, src_ptr, size);
}

void TensorCopySync(const Tensor &src, Tensor *dst) {
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_ptr = src.data<void>();
  auto dst_ptr = dst->mutable_data(src.type());
  auto size = src.numel() * SizeOfType(src.type());
  memory::Copy(dst_ptr, src_ptr, size);
}

template <typename Predicate>
struct AnyDTypeVisitor {
  Predicate predicate_;
  const Tensor &tensor_;
  Tensor *out_;

  AnyDTypeVisitor(Predicate predicate, const Tensor &tensor, Tensor *out)
      : predicate_(predicate), tensor_(tensor), out_(out) {}

  template <typename T>
  void operator()() const {
    //    auto t = EigenVector<T>::Flatten(tensor_);
    //    auto o = EigenScalar<bool>::From(*out_);
    // return any of predicate_(t) is true.
    //    o.device(*ctx_.eigen_device()) = predicate_(t).any();
  }
};

struct ContainsNANPredicate {
  template <typename T>
  auto operator()(const T &eigen_vec) const
      -> decltype(std::declval<T>().isnan()) {
    // Cast eigen_vector to vector of bool. true if is inf.
    return eigen_vec.isnan();
  }
};

struct ContainsInfPredicate {
  template <typename T>
  auto operator()(const T &eigen_vec) const
      -> decltype(std::declval<T>().isinf()) {
    // Cast eigen_vector to vector of bool. true if is inf.
    return eigen_vec.isinf();
  }
};

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void **buf, Tensor *tensor)
      : buf_(buf), tensor_(tensor) {}

  template <typename T>
  void operator()() {
    *buf_ = tensor_->mutable_data<T>();
  }

  void **buf_;
  Tensor *tensor_;
};

}  // namespace framework
}  // namespace paddle_mobile
