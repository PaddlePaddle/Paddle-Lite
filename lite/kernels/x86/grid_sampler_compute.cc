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

#include "lite/kernels/x86/grid_sampler_compute.h"
#include <string>
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = lite::fluid::EigenTensor<T, D, MajorType, IndexType>;

using Array4 = Eigen::DSizes<int64_t, 4>;

template <typename T>
inline bool IsInBound(T x, T y, T x_max, T y_max) {
  return !(x < static_cast<T>(0) || x > x_max || y < static_cast<T>(0) ||
           y > y_max);
}

template <typename T>
void Unnormalize(const X86Context& ctx,
                 Tensor* grid_slice,
                 const int max_val,  // height-1 or width-1
                 bool align_corners) {
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);

  if (!align_corners) {
    auto factor = static_cast<T>((max_val + 1) * 0.5);
    grid_slice_t.device(place) =
        (grid_slice_t + static_cast<T>(1)) * factor - static_cast<T>(0.5);
  } else {
    auto factor = static_cast<T>(max_val * 0.5);
    grid_slice_t.device(place) = (grid_slice_t + static_cast<T>(1)) * factor;
  }
}

template <typename T>
void Clip(const X86Context& ctx,
          Tensor* grid_slice,
          const int max_val,  // height-1 or width-1
          bool align_corners,
          std::string padding_mode) {
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
  auto grid_slice_t = EigenTensor<T, 3>::From(*grid_slice);
  if (padding_mode == "border") {
    grid_slice_t.device(place) = grid_slice_t.cwiseMax(static_cast<T>(0))
                                     .cwiseMin(static_cast<T>(max_val));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      auto double_range = static_cast<T>(max_val * 2);
      auto grid_abs = grid_slice_t.abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      grid_slice_t.device(place) = extra.cwiseMin(double_range - extra);
    } else {
      auto double_range = static_cast<T>((max_val + 1) * 2);
      auto grid_abs = (grid_slice_t + static_cast<T>(0.5)).abs();
      auto extra = grid_abs - (grid_abs / double_range).floor() * double_range;
      grid_slice_t.device(place) =
          extra.cwiseMin(double_range - extra) - static_cast<T>(0.5);
      grid_slice_t.device(place) = grid_slice_t.cwiseMax(static_cast<T>(0))
                                       .cwiseMin(static_cast<T>(max_val));
    }
  }
}

template <class T>
void CalcGridLocations(const X86Context& ctx,
                       const Tensor& grid,
                       const int in_h,
                       const int in_w,
                       bool align_corners,
                       std::string padding_mode,
                       Tensor* grid_x,
                       Tensor* grid_y) {
  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];

  // split grid with shape (n, h, w, 2) into (x, y) by the 3rd Dim
  DDim grid_dim{{n, out_h, out_w}};
  grid_x->Resize(grid_dim);
  grid_y->Resize(grid_dim);
  T* grid_x_data = grid_x->template mutable_data<T>();
  T* grid_y_data = grid_y->template mutable_data<T>();
  const T* grid_data = grid.data<T>();
  for (int i = 0; i < n * out_h * out_w; i++) {
    grid_x_data[i] = grid_data[2 * i];
    grid_y_data[i] = grid_data[(2 * i) + 1];
  }

  Unnormalize<T>(ctx, grid_x, in_w - 1, align_corners);
  Unnormalize<T>(ctx, grid_y, in_h - 1, align_corners);

  Clip<T>(ctx, grid_x, in_w - 1, align_corners, padding_mode);
  Clip<T>(ctx, grid_y, in_h - 1, align_corners, padding_mode);
}

template <typename T>
void GetGridPointValue(const Tensor& input,
                       Tensor* output,
                       const Tensor& x,
                       const Tensor& y) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int in_h = input.dims()[2];
  const int in_w = input.dims()[3];
  const int out_h = x.dims()[1];
  const int out_w = x.dims()[2];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto output_t =
      EigenTensor<T, 4>::From(*output).setConstant(static_cast<T>(0));
  auto input_t = EigenTensor<T, 4>::From(input);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < out_h; k++) {
      for (int l = 0; l < out_w; l++) {
        if (IsInBound(x_t(i, k, l),
                      y_t(i, k, l),
                      static_cast<T>(in_w - 1),
                      static_cast<T>(in_h - 1))) {
          for (int j = 0; j < c; j++) {
            output_t(i, j, k, l) =
                input_t(i,
                        j,
                        static_cast<int>(round(y_t(i, k, l))),
                        static_cast<int>(round(x_t(i, k, l))));
          }
        }
      }
    }
  }
}

template <typename T>
void AllNeigbors(const X86Context& ctx,
                 const Tensor& input,
                 Tensor* grid_x,
                 Tensor* grid_y,
                 Tensor* x_w,
                 Tensor* x_e,
                 Tensor* y_n,
                 Tensor* y_s,  // positions
                 Tensor* d_w,
                 Tensor* d_e,
                 Tensor* d_n,
                 Tensor* d_s,  // distance
                 Tensor* v_wn,
                 Tensor* v_en,
                 Tensor* v_ws,
                 Tensor* v_es) {  // values
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();

  const int c = input.dims()[1];
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  // calculate coords of 4 corner points
  DDim dim{{n, out_h, out_w}};
  x_w->Resize(dim);
  x_e->Resize(dim);
  y_n->Resize(dim);
  y_s->Resize(dim);
  x_w->template mutable_data<T>();
  x_e->template mutable_data<T>();
  y_n->template mutable_data<T>();
  y_s->template mutable_data<T>();
  auto x_w_t = EigenTensor<T, 3>::From(*x_w);
  auto x_e_t = EigenTensor<T, 3>::From(*x_e);
  auto y_n_t = EigenTensor<T, 3>::From(*y_n);
  auto y_s_t = EigenTensor<T, 3>::From(*y_s);

  auto grid_x_t = EigenTensor<T, 3>::From(*grid_x);
  auto grid_y_t = EigenTensor<T, 3>::From(*grid_y);

  x_w_t.device(place) = grid_x_t.floor();
  x_e_t.device(place) = x_w_t + static_cast<T>(1);
  y_n_t.device(place) = grid_y_t.floor();
  y_s_t.device(place) = y_n_t + static_cast<T>(1);

  // calculate distances to 4 sides
  d_w->Resize(dim);
  d_e->Resize(dim);
  d_n->Resize(dim);
  d_s->Resize(dim);
  d_w->template mutable_data<T>();
  d_e->template mutable_data<T>();
  d_n->template mutable_data<T>();
  d_s->template mutable_data<T>();
  auto d_w_t = EigenTensor<T, 3>::From(*d_w);
  auto d_e_t = EigenTensor<T, 3>::From(*d_e);
  auto d_n_t = EigenTensor<T, 3>::From(*d_n);
  auto d_s_t = EigenTensor<T, 3>::From(*d_s);
  d_w_t.device(place) = grid_x_t - x_w_t;
  d_e_t.device(place) = x_e_t - grid_x_t;
  d_n_t.device(place) = grid_y_t - y_n_t;
  d_s_t.device(place) = y_s_t - grid_y_t;

  // calc 4 corner points value
  DDim v_dim{{n, c, out_h, out_w}};
  v_wn->Resize(v_dim);
  v_en->Resize(v_dim);
  v_ws->Resize(v_dim);
  v_es->Resize(v_dim);
  v_wn->template mutable_data<T>();
  v_en->template mutable_data<T>();
  v_ws->template mutable_data<T>();
  v_es->template mutable_data<T>();
  GetGridPointValue<T>(input, v_wn, *x_w, *y_n);
  GetGridPointValue<T>(input, v_en, *x_e, *y_n);
  GetGridPointValue<T>(input, v_ws, *x_w, *y_s);
  GetGridPointValue<T>(input, v_es, *x_e, *y_s);
}

template <typename T>
void BilinearInter(const X86Context& ctx,
                   const Tensor& input,
                   Tensor* grid_x,
                   Tensor* grid_y,
                   Tensor* out) {
  auto place = lite::fluid::EigenDeviceType<TARGET(kX86)>();
  const int n = grid_x->dims()[0];
  const int out_h = grid_x->dims()[1];
  const int out_w = grid_x->dims()[2];
  const int c = input.dims()[1];

  Tensor x_w, x_e, y_n, y_s;
  Tensor d_w, d_e, d_n, d_s;
  Tensor v_wn, v_en, v_ws, v_es;

  AllNeigbors<T>(ctx,
                 input,
                 grid_x,
                 grid_y,
                 &x_w,
                 &x_e,
                 &y_n,
                 &y_s,
                 &d_w,
                 &d_e,
                 &d_n,
                 &d_s,
                 &v_wn,
                 &v_en,
                 &v_ws,
                 &v_es);

  auto d_w_t = EigenTensor<T, 3>::From(d_w);
  auto d_e_t = EigenTensor<T, 3>::From(d_e);
  auto d_n_t = EigenTensor<T, 3>::From(d_n);
  auto d_s_t = EigenTensor<T, 3>::From(d_s);

  auto d_w_scaled_t =
      d_w_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_e_scaled_t =
      d_e_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_n_scaled_t =
      d_n_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto d_s_scaled_t =
      d_s_t.reshape(Array4(n, 1, out_h, out_w)).broadcast(Array4(1, c, 1, 1));
  auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
  auto v_en_t = EigenTensor<T, 4>::From(v_en);
  auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
  auto v_es_t = EigenTensor<T, 4>::From(v_es);
  auto output_t = EigenTensor<T, 4>::From(*out);
  // bilinear interpolaetion by 4 corner points
  output_t.device(place) = v_wn_t * d_e_scaled_t * d_s_scaled_t +
                           v_en_t * d_w_scaled_t * d_s_scaled_t +
                           v_ws_t * d_e_scaled_t * d_n_scaled_t +
                           v_es_t * d_w_scaled_t * d_n_scaled_t;
}

template <class T>
void GridSamplerCompute<T>::Run() {
#ifndef WIN32
  auto& param = this->Param<param_t>();
  auto& context = ctx_->As<X86Context>();
  auto* input = param.x;
  auto* grid = param.grid;
  auto* output = param.out;
  const std::string padding_mode = param.padding_mode;
  const std::string mode = param.mode;
  const bool align_corners = param.align_corners;

  auto input_dims = input->dims();
  const int in_h = input_dims[2];
  const int in_w = input_dims[3];

  output->template mutable_data<T>();
  lite::x86::math::SetConstant<TARGET(kX86), T> set_zero;
  set_zero(context, output, static_cast<T>(0));

  Tensor grid_x, grid_y;
  CalcGridLocations<T>(context,
                       *grid,
                       in_h,
                       in_w,
                       align_corners,
                       padding_mode,
                       &grid_x,
                       &grid_y);
  if (mode == "bilinear") {
    BilinearInter<T>(context, *input, &grid_x, &grid_y, output);
  } else if (mode == "nearest") {
    auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
    auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
    grid_x_t = grid_x_t.round();
    grid_y_t = grid_y_t.round();
    GetGridPointValue<T>(*input, output, grid_x, grid_y);
  }
#else
  LOG(FATAL) << "Error: This model is not supported on Windows Os yet, because "
                "grid_sample op is not supported on windows Paddle-Lite, "
                "please update your Paddle-Lite version.";
#endif
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(grid_sampler,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::GridSamplerCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Grid", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
