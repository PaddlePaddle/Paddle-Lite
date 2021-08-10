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
#include <vector>
#include "lite/backends/x86/cpu_info.h"
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/jit/helper.h"
#include "lite/backends/x86/jit/kernel_base.h"
#include "lite/backends/x86/jit/kernels.h"
#include "lite/backends/x86/math/cpu_vec.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = lite::fluid::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <lite::TargetType Target, typename T, bool is_test>
void SoftmaxEigen(const lite::Context<Target>& context,
                  const int axis_dim,
                  const lite::Tensor* X,
                  lite::Tensor* Y) {
  constexpr int kBatchDim = 0;
  constexpr int kClassDim = 1;

  auto logits = EigenMatrix<T>::From(*X);
  auto softmax = EigenMatrix<T>::From(*Y);

  const int batch_size = logits.dimension(kBatchDim);
  const int num_classes = logits.dimension(kClassDim);
  const int num_remain = num_classes / axis_dim;

  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);
  Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
  Eigen::DSizes<int, 2> one_axis(1, axis_dim);

  auto shifted_logits = (logits -
                         logits.maximum(along_class)
                             .eval()
                             .reshape(batch_by_one)
                             .broadcast(one_by_class))
                            .unaryExpr(ValueClip<T>());

  softmax.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      shifted_logits.exp();
  softmax.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      (softmax *
       softmax.reshape(batch_axis_remain)
           .sum(along_class)
           .inverse()
           .eval()
           .broadcast(one_axis));
}

template <lite::TargetType Target, typename T, bool is_test, typename Enable>
void SoftmaxFunctor<Target, T, is_test, Enable>::operator()(
    const lite::Context<Target>& context,
    const int axis_dim,
    const lite::Tensor* X,
    lite::Tensor* Y) {
  SoftmaxEigen<lite::Context<Target>, T, is_test>(context, axis_dim, X, Y);
}

template <lite::TargetType Target>
using enable_if_CPU = typename std::enable_if<
    std::is_same<lite::Context<Target>, lite::X86Context>::value>::type;

template <lite::TargetType Target, typename T, bool is_test>
class SoftmaxFunctor<Target, T, is_test, enable_if_CPU<Target>> {
 public:
  void operator()(const lite::Context<Target>& context,
                  const int axis_dim,
                  const lite::Tensor* X,
                  lite::Tensor* Y) {
    const auto& in_dims = X->dims();
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int num_classes = in_dims[kClassDim];
    const int batch_size = in_dims[kBatchDim];
    const int num_remain = num_classes / axis_dim;

    if (num_remain == 1 && lite::x86::MayIUse(lite::x86::avx)) {
      const T* in_data = X->template data<T>();
      auto* out_data = Y->template mutable_data<T>();
      for (int bs = 0; bs < batch_size; ++bs) {
        T max_val = *std::max_element(in_data, in_data + num_classes);
        max_val *= static_cast<T>(-1);
        vec_add_bias<T, lite::x86::avx>(
            num_classes, max_val, in_data, out_data);
        vec_clip<T, lite::x86::avx>(
            num_classes, static_cast<T>(-64), out_data, out_data);
        vec_exp<T>(num_classes, out_data, out_data);

        T sum = 0;
        vec_sum<T, lite::x86::avx>(num_classes, out_data, &sum);
        sum = static_cast<T>(1) / sum;
        vec_scal<T, lite::x86::avx>(num_classes, sum, out_data, out_data);

        in_data += num_classes;
        out_data += num_classes;
      }
    } else {
      SoftmaxEigen<Target, T, is_test>(context, axis_dim, X, Y);
    }
  }
};

template <lite::TargetType Target>
class SoftmaxFunctor<Target, float, true, enable_if_CPU<Target>> {
 public:
  void operator()(const lite::Context<Target>& context,
                  const int axis_dim,
                  const lite::Tensor* X,
                  lite::Tensor* Y) {
    const auto& in_dims = X->dims();
    const float* in_data = X->data<float>();
    float* out_data = Y->mutable_data<float>();
    const int kBatchDim = 0;
    const int kClassDim = 1;
#ifdef PADDLE_WITH_MKLML
    // 2D data. Batch x C
    auto compute_softmax =
        lite::jit::KernelFuncs<lite::jit::SoftmaxTuple<float>,
                               fluid::CPUPlace>::Cache()
            .At(in_dims[kClassDim]);
    compute_softmax(in_data,
                    out_data,
                    in_dims[kClassDim],
                    in_dims[kBatchDim],
                    in_dims[kClassDim] / axis_dim);
#else
    const int batch_size = in_dims[kBatchDim];
    const int length = in_dims[kClassDim];
    const int stride = in_dims[kClassDim] / axis_dim;
    for (int bs = 0; bs < batch_size; ++bs) {
      // get max value of input data
      float in_max = -FLT_MAX;
      for (int i = 0; i < length; ++i) {
        in_max = (std::max)(in_max, in_data[i]);
      }
      // y = exp(x - in_max)
      for (int i = 0; i < length; ++i) {
        out_data[i] = static_cast<float>(std::exp(in_data[i] - in_max));
      }
      // y = y / sum(y[i], y[i + stride], y[i + stride + stride] ...)
      for (int i = 0; i < stride; ++i) {
        float sum = 0.f;
        for (int j = 0; j < axis_dim; ++j) {
          sum += out_data[i + j * stride];
        }
        for (int j = 0; j < axis_dim; ++j) {
          out_data[i + j * stride] /= sum;
        }
      }
      in_data += length;
      out_data += length;
    }
#endif
  }
};

template <lite::TargetType Target, typename T>
void SoftmaxGradEigen(const lite::Context<Target>& context,
                      const int axis_dim,
                      const lite::Tensor* y,
                      const lite::Tensor* y_grad,
                      lite::Tensor* x_grad) {
  auto softmax = EigenMatrix<T>::From(*y);
  auto softmax_grad = EigenMatrix<T>::From(*y_grad);
  auto logits_grad = EigenMatrix<T>::From(*x_grad);

  constexpr int kBatchDim = 0;
  constexpr int kClassDim = 1;

  const int batch_size = softmax.dimension(kBatchDim);
  const int num_classes = softmax.dimension(kClassDim);
  const int num_remain = num_classes / axis_dim;

  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);
  Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
  Eigen::DSizes<int, 2> one_axis(1, axis_dim);

  auto dot = (softmax * softmax_grad)
                 .reshape(batch_axis_remain)
                 .sum(along_class)
                 .eval()
                 .broadcast(one_axis);
  // logits_grad.device(*context.eigen_device()) = (softmax_grad - dot) *
  // softmax;
  logits_grad.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      (softmax_grad - dot) * softmax;
}

template <lite::TargetType Target, typename T, typename Enable>
void SoftmaxGradFunctor<Target, T, Enable>::operator()(
    const lite::Context<Target>& context,
    const int axis_dim,
    const lite::Tensor* y,
    const lite::Tensor* y_grad,
    lite::Tensor* x_grad) {
  SoftmaxGradEigen<lite::Context<Target>, T>(
      context, axis_dim, y, y_grad, x_grad);
}

template <lite::TargetType Target, typename T>
class SoftmaxGradFunctor<Target, T, enable_if_CPU<Target>> {
 public:
  void operator()(const lite::Context<Target>& context,
                  const int axis_dim,
                  const lite::Tensor* y,
                  const lite::Tensor* y_grad,
                  lite::Tensor* x_grad) {
    auto out_dims = y->dims();
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    const int num_classes = out_dims[kClassDim];
    const int batch_size = out_dims[kBatchDim];
    const int num_remain = num_classes / axis_dim;

    if (num_remain == 1 && lite::x86::MayIUse(lite::x86::avx)) {
      const T* out_data = y->template data<T>();
      const T* out_grad = y_grad->template data<T>();
      T* in_grad = x_grad->template mutable_data<T>();
      for (int bs = 0; bs < batch_size; ++bs) {
        T scalar;
        vec_mul_reduce<T, lite::x86::avx>(
            num_classes, out_grad, out_data, &scalar);
        scalar *= static_cast<T>(-1);
        vec_add_bias<T, lite::x86::avx>(num_classes, scalar, out_grad, in_grad);
        vec_mul<T, lite::x86::avx>(num_classes, out_data, in_grad, in_grad);
        out_data += num_classes;
        out_grad += num_classes;
        in_grad += num_classes;
      }
    } else {
      SoftmaxGradEigen<Target, T>(context, axis_dim, y, y_grad, x_grad);
    }
  }
};

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
