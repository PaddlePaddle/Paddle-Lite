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

#include "lite/kernels/xpu/mul_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MulCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& origin_x = *param.x;
  auto& origin_y = *param.y;
  auto& x_dims = origin_x.dims();
  auto& y_dims = origin_y.dims();
  Tensor x_matrix, y_matrix;
  if (x_dims.size() > 2) {
    x_matrix = ReshapeToMatrix(origin_x, param.x_num_col_dims);
  } else {
    x_matrix = origin_x;
  }
  if (y_dims.size() > 2) {
    y_matrix = ReshapeToMatrix(origin_y, param.y_num_col_dims);
  } else {
    y_matrix = origin_y;
  }
  int m = x_matrix.dims()[0];
  int k = x_matrix.dims()[1];
  int n = y_matrix.dims()[1];

  int r = xdnn::fc_fusion<float, float, float, int16_t>(
      ctx.GetRawContext(),                              // ctx
      x_matrix.data<float>(),                           // x
      y_matrix.data<float>(),                           // w
      param.output->mutable_data<float>(TARGET(kXPU)),  // y
      m,                                                // m
      n,                                                // n
      k,                                                // k
      false,                                            // x_trans
      false,                                            // w_trans
      nullptr,                                          // x_maxptr
      nullptr,                                          // w_maxptr
      nullptr,                                          // y_maxptr
      k,                                                // ldx
      n,                                                // ldw
      n,                                                // ldy
      1.0f,                                             // alpha
      0.0f,                                             // beta
      nullptr,                                          // bias
      xdnn::Activation_t::LINEAR);                      // act

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mul, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::MulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
