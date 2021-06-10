// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/x86/gru_unit_compute.h"
#include "lite/backends/x86/math/blas.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = lite::fluid::EigenMatrix<T, MajorType, IndexType>;

template <class T>
void GRUUnitCompute<T>::Run() {
#ifndef WIN32
  auto& param = this->Param<param_t>();
  auto& context = ctx_->As<X86Context>();
  auto* input = param.input;
  auto* hidden_prev = param.hidden_prev;
  auto* weight = param.weight;
  auto* bias = param.bias;
  auto* gate = param.gate;
  gate->template mutable_data<T>();
  auto* reset_hidden_prev = param.reset_hidden_prev;
  reset_hidden_prev->template mutable_data<T>();
  auto* hidden = param.hidden;
  hidden->template mutable_data<T>();

  int batch_size = input->dims()[0];
  int frame_size = hidden_prev->dims()[1];

  auto x = EigenMatrix<T>::From(*input);
  auto h_p = EigenMatrix<T>::From(*hidden_prev);
  auto g = EigenMatrix<T>::From(*gate);
  auto r_h_p = EigenMatrix<T>::From(*reset_hidden_prev);
  auto h = EigenMatrix<T>::From(*hidden);
  const auto& place = lite::fluid::EigenDeviceType<lite::TargetType::kX86>();

  if (bias) {
    auto b = EigenMatrix<T>::From(*bias);
    g.device(place) = x +
                      b.reshape(Eigen::array<int, 2>({{1, frame_size * 3}}))
                          .broadcast(Eigen::array<int, 2>({{batch_size, 1}}));
  } else {
    g.device(place) = x;
  }

  // calculate unactivated gate outputs
  const T* hidden_prev_data = hidden_prev->template data<T>();
  const T* weight_data = weight->template data<T>();
  T* gate_data = gate->template mutable_data<T>();
  T* reset_hidden_prev_data = reset_hidden_prev->template mutable_data<T>();
  auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(context);
  blas.GEMM(false,
            false,
            batch_size,
            2 * frame_size,
            frame_size,
            1,
            hidden_prev_data,
            frame_size,
            weight_data,
            frame_size * 2,
            1,
            gate_data,
            frame_size * 3);

  // calculate activited gate
  Eigen::array<int, 2> extents{{batch_size, frame_size}};
  Eigen::array<int, 2> u_offsets{{0, 0}};
  ActCompute(param.gate_activation,
             place,
             g.slice(u_offsets, extents),
             g.slice(u_offsets, extents));
  auto u = g.slice(u_offsets, extents);  // update gate
  Eigen::array<int, 2> r_offsets{{0, frame_size}};
  ActCompute(param.gate_activation,
             place,
             g.slice(r_offsets, extents),
             g.slice(r_offsets, extents));
  auto r = g.slice(r_offsets, extents);  // reset gate
  r_h_p.device(place) = r * h_p;         // reset previous hidden state
  blas.GEMM(false,
            false,
            batch_size,
            frame_size,
            frame_size,
            1,
            reset_hidden_prev_data,
            frame_size,
            weight_data + frame_size * frame_size * 2,
            frame_size,
            1,
            gate_data + frame_size * 2,
            frame_size * 3);

  Eigen::array<int, 2> c_offsets{{0, frame_size * 2}};
  ActCompute(param.activation,
             place,
             g.slice(c_offsets, extents),
             g.slice(c_offsets, extents));
  auto c = g.slice(c_offsets, extents);  // output candidate

  // calculate final output
  if (param.origin_mode) {
    h.device(place) = c + u * (h_p - c);  // (1 - u) * c + u * h_p
  } else {
    h.device(place) = u * (c - h_p) + h_p;  // u * c + (1 - u) * h_p
  }
#else
  LOG(FATAL) << "Error: this model is not supported on Windows Os yet, because "
                "gru_unit kernel is not supported on Windows Paddle-Lite, "
                "please update your Paddle-Lite version.";
#endif
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gru_unit,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::GRUUnitCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("HiddenPrev", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Gate", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("ResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
