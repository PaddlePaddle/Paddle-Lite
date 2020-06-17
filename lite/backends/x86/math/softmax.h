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
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <lite::TargetType Target,
          typename T,
          bool is_test,
          typename Enable = void>
class SoftmaxFunctor {
 public:
  void operator()(const lite::Context<Target>& context,
                  const int axis_dim,
                  const lite::Tensor* X,
                  lite::Tensor* Y);
};

template <lite::TargetType Target, typename T, typename Enable = void>
class SoftmaxGradFunctor {
 public:
  void operator()(const lite::Context<Target>& context,
                  const int axis_dim,
                  const lite::TensorLite* y,
                  const lite::TensorLite* y_grad,
                  lite::TensorLite* x_grad);
};

//#ifdef PADDLE_WITH_CUDA
// template <typename T>
// class SoftmaxCUDNNFunctor {
// public:
//  void operator()(const platform::CUDADeviceContext& context,
//                  const lite::TensorLite* X, lite::TensorLite* Y);
//};
//
// template <typename T>
// class SoftmaxGradCUDNNFunctor {
// public:
//  void operator()(const platform::CUDADeviceContext& context,
//                  const lite::TensorLite* Y, const lite::TensorLite* y_grad,
//                  lite::TensorLite* x_grad);
//};
//
//#endif

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
