// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/cos_sim_compute.h"
#include <cmath>
#include "lite/backends/x86/fluid/for_range.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, bool same_row>
struct CosSimFunctor {
  CosSimFunctor(const T* x, const T* y, T* x_norm, T* y_norm, T* z, size_t cols)
      : x_norm_(x_norm), y_norm_(y_norm), x_(x), y_(y), z_(z), cols_(cols) {}

  inline void operator()(size_t row_id) const {
    auto* x = x_ + cols_ * row_id;
    T xx = 0, xy = 0, yy = 0;
    if (same_row) {
      auto* y = y_ + cols_ * row_id;
      T tep_x, tep_y;
      for (size_t i = 0; i < cols_; ++i) {
        tep_x = x[i];
        tep_y = y[i];
        xx += tep_x * tep_x;
        yy += tep_y * tep_y;
        xy += tep_x * tep_y;
      }
      xx = sqrt(xx);
      yy = sqrt(yy);
      y_norm_[row_id] = yy;
      x_norm_[row_id] = xx;
      z_[row_id] = xy / (xx * yy);
    } else {  // This can be wrote in a better way.
      T tep_x, tep_y;
      for (size_t i = 0; i < cols_; ++i) {
        tep_x = x[i];
        tep_y = y_[i];
        xx += tep_x * tep_x;
        yy += tep_y * tep_y;
        xy += tep_x * tep_y;
      }
      xx = sqrt(xx);
      yy = sqrt(yy);
      if (row_id == 0) y_norm_[0] = yy;
      x_norm_[row_id] = xx;
      z_[row_id] = xy / (xx * yy);
    }
  }

  T* x_norm_;
  T* y_norm_;
  const T* x_;
  const T* y_;
  T* z_;
  const size_t cols_;
};

template <typename T>
void CosSimCompute<T>::Run() {
  auto& ctx = ctx_->template As<HostContext>();
  auto& param = this->template Param<param_t>();
  auto* x = param.x;
  auto* y = param.y;
  auto* out = param.out;
  auto* x_norm = param.x_norm;
  auto y_norm = param.y_norm;

  size_t rows_x = x->dims()[0];
  size_t rows_y = y->dims()[0];
  size_t cols = static_cast<size_t>(x->numel() / rows_x);

  if (rows_x == rows_y) {
    CosSimFunctor<T, true> functor(x->template data<T>(),
                                   y->template data<T>(),
                                   x_norm->template mutable_data<T>(),
                                   y_norm->template mutable_data<T>(),
                                   out->template mutable_data<T>(),
                                   cols);
    fluid::ForRange<TARGET(kHost)> for_range(ctx, rows_x);
    for_range(functor);
  } else {
    CosSimFunctor<T, false> functor(x->template data<T>(),
                                    y->template data<T>(),
                                    x_norm->template mutable_data<T>(),
                                    y_norm->template mutable_data<T>(),
                                    out->template mutable_data<T>(),
                                    cols);
    fluid::ForRange<TARGET(kHost)> for_range(ctx, rows_x);
    for_range(functor);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(cos_sim,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::CosSimCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("XNorm", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("YNorm", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
