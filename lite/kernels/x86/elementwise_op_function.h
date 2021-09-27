/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <vector>
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/fluid/for_range.h"
#include "lite/backends/x86/fluid/transform.h"
#include "lite/backends/x86/math/math_function.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/variant.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

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
 *
 * New parameter: *mid_flag* is added to solve m*n*k & m*1*k
 * broadcast cases.
 */
inline void get_mid_dims(const lite::DDim &x_dims,
                         const lite::DDim &y_dims,
                         const int axis,
                         int *pre,
                         int *n,
                         int *post,
                         int *mid_flag = NULL) {
  *pre = 1;
  *n = 1;
  *post = 1;
  if (mid_flag != NULL) {
    *mid_flag = 0;
    for (int i = 0; i < axis; ++i) {
      (*pre) *= x_dims[i];
    }
    for (size_t i = 0; i < y_dims.size(); ++i) {
      if (x_dims[i + axis] != y_dims[i]) {
        CHECK_EQ(y_dims[i] == 1 || x_dims[i + axis] == 1, true)
            << "Broadcast y or x dimension is not 1.";
        *mid_flag = 1;
        return;
      }
      (*n) *= y_dims[i];
    }
    for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
      (*post) *= x_dims[i];
    }
  } else {  // for fused_elementwise_activation_op. keep the old version.
    for (int i = 0; i < axis; ++i) {
      (*pre) *= x_dims[i];
    }

    for (size_t i = 0; i < y_dims.size(); ++i) {
      CHECK_EQ(x_dims[i + axis], y_dims[i]) << "Broadcast dimension mismatch.";
      (*n) *= y_dims[i];
    }

    for (size_t i = axis + y_dims.size(); i < x_dims.size(); ++i) {
      (*post) *= x_dims[i];
    }
  }
}

inline lite::DDim trim_trailing_singular_dims(const lite::DDim &dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size == dims.size()) return dims;
  std::vector<int64_t> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (size_t i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return lite::DDim();
  }
  lite::DDim actual_dims = lite::DDim(trim_dims);
  return actual_dims;
}

template <typename T, lite::TargetType Target>
class RowwiseTransformIterator;

template <typename T, lite::TargetType Target>
class MidWiseTransformIterator;

// NOTE(dzhwinter): ptrdiff_t in iterator is deperecated in c++17
template <typename T>
class RowwiseTransformIterator<T, lite::TargetType::kX86>
    : public std::iterator<std::random_access_iterator_tag,
                           T,
                           std::ptrdiff_t,
                           T *,
                           T &> {
 public:
  RowwiseTransformIterator(const T *ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

  RowwiseTransformIterator<T, lite::TargetType::kX86> &operator++() {
    ++i_;
    if (UNLIKELY(i_ == n_)) {
      i_ = 0;
    }
    return *this;
  }

  RowwiseTransformIterator<T, lite::TargetType::kX86> &operator+(int n) {
    while (n-- > 0) {
      ++i_;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }

    return *this;
  }

  bool operator==(
      const RowwiseTransformIterator<T, lite::TargetType::kX86> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(
      const RowwiseTransformIterator<T, lite::TargetType::kX86> &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int i_;
  int64_t n_;
};

template <typename T>
class MidWiseTransformIterator<T, lite::TargetType::kX86>
    : public std::iterator<std::random_access_iterator_tag,
                           T,
                           std::ptrdiff_t,
                           T *,
                           T &> {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T, lite::TargetType::kX86> &operator++() {
    ++j_;
    if (UNLIKELY(j_ == post_)) {
      ++i_;
      j_ = 0;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }
    return *this;
  }

  MidWiseTransformIterator<T, lite::TargetType::kX86> &operator+(int n) {
    while (n-- > 0) {
      ++j_;
      if (UNLIKELY(j_ == post_)) {
        ++i_;
        j_ = 0;
        if (UNLIKELY(i_ == n_)) {
          i_ = 0;
        }
      }
    }
    return *this;
  }

  bool operator==(
      const MidWiseTransformIterator<T, lite::TargetType::kX86> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(
      const MidWiseTransformIterator<T, lite::TargetType::kX86> &rhs) const {
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

template <typename Functor,
          typename T,
          lite::TargetType Target,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const lite::Tensor *x,
                   const lite::Tensor *y,
                   lite::Tensor *z,
                   const lite::Context<Target> &ctx,
                   Functor func,
                   const bool is_xsize_larger = true)
      : x_(x->template data<T>()),
        y_(y->template data<T>()),
        z_(z->mutable_data<OutType>()),
        nx_(x->numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y->numel();
    }
  }

  inline void Run() const {
    lite::fluid::Transform<Target> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    lite::fluid::Transform<Target> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            RowwiseTransformIterator<T, Target>(y_, n),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            RowwiseTransformIterator<T, Target>(x_, n),
            z_,
            func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    lite::fluid::Transform<Target> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            MidWiseTransformIterator<T, Target>(y_, n, post),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            MidWiseTransformIterator<T, Target>(x_, n, post),
            z_,
            func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const lite::Context<Target> &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

inline void GetBroadcastDimsArrays(const DDim &x_dims,
                                   const DDim &y_dims,
                                   int *x_dims_array,
                                   int *y_dims_array,
                                   int *out_dims_array,
                                   const int max_dim,
                                   const int axis) {
  CHECK_GE(axis, 0) << "Axis should be great than or equal to 0.";
  CHECK_LT(axis, max_dim) << "Axis should be less than max(x_dim, y_dim).";

  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    for (int i = 0; i < x_dims.size(); i++) x_dims_array[i] = x_dims[i];
    for (int i = 0; i < y_dims.size(); i++)
      *(y_dims_array + axis + i) = y_dims[i];
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    for (int i = 0; i < x_dims.size(); i++)
      *(x_dims_array + axis + i) = x_dims[i];
    for (int i = 0; i < y_dims.size(); i++) *(y_dims_array + i) = y_dims[i];
  }

  for (int i = 0; i < max_dim; i++) {
    CHECK_EQ(x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
                 y_dims_array[i] <= 1,
             true)
        << "Broadcast dimension mismatch. Operands could not be broadcast.";

    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

inline int GetElementwiseIndex(const int *x_dims_array,
                               const int max_dim,
                               const int *index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

inline void UpdateElementwiseIndexArray(const int *out_dims_array,
                                        const int max_dim,
                                        int *index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(const Tensor *x,
                               const Tensor *y,
                               Tensor *z,
                               int *x_dims_array,
                               int *y_dims_array,
                               int *out_dims_array,
                               int max_dim,
                               Functor func,
                               const bool is_xsize_larger = true) {
  std::vector<int> index_array(max_dim, 0);
  const T *x_data = x->data<T>();
  const T *y_data = y->data<T>();
  CHECK_EQ(x_data != nullptr, true) << "The input X should not be empty.";
  CHECK_EQ(y_data != nullptr, true) << "The input Y should not be empty.";

  OutType *out_data = z->mutable_data<OutType>();
  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (is_xsize_larger) {
      out_data[out_index] = func(x_data[x_index], y_data[y_index]);
    } else {
      out_data[out_index] = func(y_data[y_index], x_data[x_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonElementwiseBroadcastForward(const Tensor *x,
                                       const Tensor *y,
                                       Tensor *z,
                                       const DDim &x_dims,
                                       const DDim &y_dims,
                                       Functor func,
                                       int axis,
                                       const bool is_xsize_larger = true) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  CHECK_GE(axis, 0) << "Axis should be great than or equal to 0.";
  CHECK_LT(axis, max_dim) << "Axis should be less than max(x_dim, y_dim).";

  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  CommonForwardBroadcastCPU<Functor, T, OutType>(x,
                                                 y,
                                                 z,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 func,
                                                 is_xsize_larger);
}

template <typename Functor,
          lite::TargetType Target,
          typename T,
          typename OutType = T>
void ElementwiseComputeEx(const lite::Context<Target> &ctx,
                          const lite::Tensor *x,
                          const lite::Tensor *y,
                          int axis,
                          Functor func,
                          lite::Tensor *z) {
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }
  TransformFunctor<Functor, T, Target, OutType> functor(
      x, y, z, ctx, func, is_xsize_larger);
  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  int tmp = std::abs(static_cast<int>(x_dims.size()) -
                     static_cast<int>(y_dims.size()));
  axis = (axis == static_cast<int>(-1) ? tmp : axis);

  CHECK_GE(axis, 0) << "Axis should be great than or equal to 0.";
  CHECK_LT(axis, max_dim) << "Axis should be less than max(x_dim, y_dim).";

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    get_mid_dims(x_dims,
                 y_dims_trimed,
                 axis_trim,
                 &pre,
                 &n,
                 &post,
                 &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    get_mid_dims(y_dims,
                 x_dims_trimed,
                 axis_trim,
                 &pre,
                 &n,
                 &post,
                 &is_run_common_broadcast);
  }
  // special case for common implementation.
  // case 1: x=[2,3,1,5], y=[2,1,4,1]
  // case 2: x=[2,3,4], y=[1,1,4]
  if (is_run_common_broadcast == 1) {
    CommonElementwiseBroadcastForward<Functor, T, OutType>(
        x, y, z, x_dims, y_dims, func, axis, is_xsize_larger);
    return;
  }
  if (post == 1) {
    functor.RunRowWise(n, pre);
    return;
  } else {
    functor.RunMidWise(n, pre, post);
    return;
  }
}

// FusedElemwiseAndAct
// --- forward
template <typename T, typename CompoundFunctor, bool KeepIntermediateOut>
struct FusedElemwiseAndActNoBroadcast {
  HOSTDEVICE void operator()(size_t i) {
    T y_val = y_[i];
    T x_val = x_[i];
    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor_.GetIntermediateOut(x_val, y_val);
      intermediate_out_[i] = intermeidiate_out;
      out_[i] =
          compound_functor_.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out_[i] = compound_functor_.GetOut(x_val, y_val);
    }
  }

  const T *x_;
  const T *y_;
  CompoundFunctor compound_functor_;
  T *out_;
  T *intermediate_out_;
};

// FusedElemwiseAndActBroadcast1:
// In this case, X and Y can be reshaped to a matrix.
// For example shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5) and axis = -1 or 2,
// X can be reshaped to (6, 20) and Y can be reshaped to (1, 20)
template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CPU(const T *x,
                                             const T *y,
                                             CompoundFunctor compound_functor,
                                             int h,
                                             int w,
                                             T *out,
                                             T *intermediate_out) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      T y_val = BcastY ? y[j] : y[offset];
      T x_val = BcastY ? x[offset] : x[j];
      int64_t intermediate_out_offset;
      if (KeepIntermediateOut) {
        T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

        if (SameShapeOfIntermediateOutAndOut) {
          // for the case of f1(f2(x, y))
          intermediate_out_offset = offset;
        } else if (BcastY) {
          intermediate_out_offset = j;
        } else {
          intermediate_out_offset = offset;
        }

        intermediate_out[intermediate_out_offset] = intermeidiate_out;
        out[offset] =
            compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
      } else {
        out[offset] = compound_functor.GetOut(x_val, y_val);
      }
    }
  }
}

// FusedElemwiseAndActBroadcast2
// In this case, X and Y can be reshaped to a matrix.
// For example shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4) and axis = 1,
// X can be reshaped to (2, 12, 5) and Y can be reshaped to (1, 12, 1)
// pre = 2, n = 12, post = 5
template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CPU(const T *x,
                                             const T *y,
                                             int pre,
                                             int n,
                                             int post,
                                             CompoundFunctor compound_functor,
                                             T *out,
                                             T *intermediate_out) {
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        T y_val = BcastY ? y[j] : y[offset];
        T x_val = BcastY ? x[offset] : x[j];
        int64_t intermediate_out_offset;

        if (KeepIntermediateOut) {
          T intermeidiate_out =
              compound_functor.GetIntermediateOut(x_val, y_val);

          if (SameShapeOfIntermediateOutAndOut) {
            // for the case of f1(f2(x, y))
            intermediate_out_offset = offset;
          } else if (BcastY) {
            intermediate_out_offset = j;
          } else {
            intermediate_out_offset = offset;
          }

          intermediate_out[intermediate_out_offset] = intermeidiate_out;
          out[offset] = compound_functor.GetOutUseIntermediateOut(
              x_val, intermeidiate_out);
        } else {
          out[offset] = compound_functor.GetOut(x_val, y_val);
        }
      }
    }
  }
}

template <lite::TargetType Target,
          typename T,
          typename CompoundFunctor,
          bool KeepIntermediateOut>
void FusedElemwiseAndActComputeNoBroadcast(const lite::Context<Target> &ctx,
                                           const lite::DDim &x_dim,
                                           const lite::Tensor &x,
                                           const lite::Tensor &y,
                                           CompoundFunctor compound_functor,
                                           lite::Tensor *out,
                                           lite::Tensor *intermediate_out) {
  size_t N = static_cast<size_t>(x_dim.production());

  lite::fluid::ForRange<Target> for_range(ctx, N);

  for_range(
      FusedElemwiseAndActNoBroadcast<T, CompoundFunctor, KeepIntermediateOut>{
          x.data<T>(),
          y.data<T>(),
          compound_functor,
          out->template mutable_data<T>(),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->template mutable_data<T>()});
}

template <lite::TargetType Target,
          typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeWithBroadcast(const lite::Context<Target> &ctx,
                                             const lite::DDim &x_dim,
                                             const lite::DDim &y_dim_untrimed,
                                             const lite::Tensor &x,
                                             const lite::Tensor &y,
                                             CompoundFunctor compound_functor,
                                             int axis,
                                             lite::Tensor *out,
                                             lite::Tensor *intermediate_out) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post);

  if (post == 1) {
    int h = pre;
    int w = n;
    FusedElemwiseAndActBroadcast1CPU<T,
                                     CompoundFunctor,
                                     BcastY,
                                     KeepIntermediateOut,
                                     SameShapeOfIntermediateOutAndOut>(
        x.data<T>(),
        y.data<T>(),
        compound_functor,
        h,
        w,
        out->template mutable_data<T>(),
        intermediate_out == nullptr
            ? nullptr
            : intermediate_out->template mutable_data<T>());

  } else {
    FusedElemwiseAndActBroadcast2CPU<T,
                                     CompoundFunctor,
                                     BcastY,
                                     KeepIntermediateOut,
                                     SameShapeOfIntermediateOutAndOut>(
        x.data<T>(),
        y.data<T>(),
        pre,
        n,
        post,
        compound_functor,
        out->template mutable_data<T>(),
        intermediate_out == nullptr
            ? nullptr
            : intermediate_out->template mutable_data<T>());
  }
}

template <lite::TargetType Target,
          typename T,
          typename CompoundFunctor,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeEx(const lite::Context<Target> &ctx,
                                  const lite::Tensor &x,
                                  const lite::Tensor &y,
                                  int axis,
                                  CompoundFunctor compound_functor,
                                  lite::Tensor *out,
                                  lite::Tensor *intermediate_out) {
  if (KeepIntermediateOut) {
    CHECK(intermediate_out) << "The save_intermediate_out is opened, "
                               "intermediate_out should not be nullptr.";
  }

  const lite::DDim &x_dim = x.dims();
  const lite::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    FusedElemwiseAndActComputeNoBroadcast<Target,
                                          T,
                                          CompoundFunctor,
                                          KeepIntermediateOut>(
        ctx, x_dim, x, y, compound_functor, out, intermediate_out);
  } else {
    // Whether the shape of Y is a continuous subsequence of X,
    // For more information please refer to the op's introduction.
    bool bcast_y = x.dims().size() >= y.dims().size();
    if (x.dims().size() == y.dims().size()) {
      for (int i = 0; i < x.dims().size(); ++i) {
        if (x.dims()[i] < y.dims()[i]) {
          bcast_y = false;
          break;
        }
      }
    }

    // z = f1(x, f2(y))
    // z = f1(f2(x, y))
    if (bcast_y) {  // Y should be broadcast.
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
      // of Y.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of X.
      FusedElemwiseAndActComputeWithBroadcast<Target,
                                              T,
                                              CompoundFunctor,
                                              true /*BcastY*/,
                                              KeepIntermediateOut,
                                              SameShapeOfIntermediateOutAndOut>(
          ctx,
          x_dim /*OutShape*/,
          y_dim,
          x,
          y,
          compound_functor,
          axis,
          out,
          intermediate_out);
    } else {
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
      // of Out.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of Y.
      FusedElemwiseAndActComputeWithBroadcast<Target,
                                              T,
                                              CompoundFunctor,
                                              false /*BcastY*/,
                                              KeepIntermediateOut,
                                              SameShapeOfIntermediateOutAndOut>(
          ctx,
          y_dim /*OutShape*/,
          x_dim,
          x,
          y,
          compound_functor,
          axis,
          out,
          intermediate_out);
    }
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
