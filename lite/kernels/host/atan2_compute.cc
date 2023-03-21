// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/host/atan2_compute.h"
#include <cmath>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
struct Atan2Out {
  using type = T;
};

template <typename T>
struct Atan2Functor {
  Atan2Functor(const T* x1,
               const T* x2,
               typename Atan2Out<T>::type* out,
               int64_t numel)
      : x1_(x1), x2_(x2), out_(out), numel_(numel) {}

  void operator()(int64_t idx) const {
    out_[idx] = static_cast<typename Atan2Out<T>::type>(
        ::atan2f(static_cast<float>(x1_[idx]), static_cast<float>(x2_[idx])));
  }

  const T* x1_;
  const T* x2_;
  typename Atan2Out<T>::type* out_;
  int64_t numel_;
};

template <typename T>
void Atan2Compute<T>::Run() {
  auto& param = this->template Param<param_t>();

  const auto* x1_data = param.X1->template data<T>();
  const auto* x2_data = param.X2->template data<T>();
  auto* out_data =
      param.Out->template mutable_data<typename Atan2Out<T>::type>();
  auto x1_numel = param.X1->numel();
  auto x2_numel = param.X2->numel();

  CHECK_LT(x1_numel, x2_numel) << "The count of elements of X1 shall not "
                               << "greater than count of elements of X2.";
  Atan2Functor<T> functor(x1_data, x2_data, out_data, x1_numel);
  for (int64_t i = 0; i < x1_numel; ++i) {
    functor(i);
  }

#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "atan2_func";
#endif
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using atan2_float = paddle::lite::kernels::host::Atan2Compute<float>;
REGISTER_LITE_KERNEL(atan2, kHost, kAny, kNCHW, atan2_float, def)
    .BindInput("X1", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("X2", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();
