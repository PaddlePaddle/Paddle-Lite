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
#include "transform.h"

#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

namespace paddle_mobile {
namespace operators {

/*
 * Out = X ⊙ Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1, 12, 1).broadcast(2, 12, 5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(6, 20, 1) * y.shape(1, 20, 1).broadcast(6, 20, 1)
 */
inline void get_mid_dims(const framework::DDim &x_dims,
                         const framework::DDim &y_dims, const int axis,
                         int *pre, int *n, int *post) {
  *pre = 1;
  *n = 1;
  *post = 1;
  // compute pre
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }

  for (int i = 0; i < y_dims.size(); ++i) {
    assert(x_dims[i + axis] == y_dims[i]);
    /// "Broadcast dimension mismatch.");
    (*n) *= y_dims[i];
  }

  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
}

/// remove dims tail 1. (4,20,1,1) -> (4,20)
inline void trim_trailing_singular_dims(framework::DDim *dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims->size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if ((*dims)[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size != dims->size()) {
    auto actual_dims = framework::vectorize(*dims);
    actual_dims.resize(actual_dims_size);
    *dims = framework::make_ddim(actual_dims);
  }
}

/// (4,20,2)+(20,): (20,) just as (20,1), when move 2 strides in last
/// dimension
/// in (4,20,2) is 2 ,
/// (20,1) move 1 stride , to fill(add) 2 element with the same number.
template <typename T>
class MidWiseTransformIterator {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T> &operator++() {
    if (post_ != 1) {
      ++j_;
      if (UNLIKELY(j_ == post_)) {
        ++i_;
        j_ = 0;
        if (UNLIKELY(i_ == n_)) {
          i_ = 0;
        }
      }
      return *this;
    } else {
      ++i_;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
      return *this;
    }
  }

  bool operator==(const MidWiseTransformIterator<T> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const MidWiseTransformIterator<T> &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int64_t i_;
  int64_t j_;
  int64_t n_;
  int64_t post_;
};

template <typename Functor, typename T, typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, Functor func)
      : x_(x->data<T>()),
        y_(y->data<T>()),
        z_(z->mutable_data<OutType>()),
        nx_(x->numel()),
        func_(func) {}

  inline void Run() const {
    math::Transform trans;
    // 同时执行func(x_, y_)传入z_。
    trans(x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunMidWise(int n, int pre, int post) const {
    math::Transform trans;
    trans(x_, x_ + nx_, MidWiseTransformIterator<T>(y_, n, post), z_, func_);
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  Functor func_;
};

template <typename Functor, typename T, typename OutType = T>
void ElementwiseComputeEx(const framework::Tensor *x,
                          const framework::Tensor *y, int axis, Functor func,
                          framework::Tensor *z) {
  TransformFunctor<Functor, T, OutType> functor(x, y, z, func);

  auto x_dims = x->dims();
  auto y_dims = y->dims();
  PADDLE_MOBILE_ENFORCE(x_dims.size() >= y_dims.size(),
                        "Rank of first input must >= rank of second input.");

  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  /// axis = -1 represent the last dimensions.
  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
  PADDLE_MOBILE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                        "Axis should be in range [0, x_dims)");
  trim_trailing_singular_dims(&y_dims);
  axis = (y_dims.size() == 0) ? x_dims.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

  functor.RunMidWise(n, pre, post);
}

}  // namespace operators
}  // namespace paddle_mobile
