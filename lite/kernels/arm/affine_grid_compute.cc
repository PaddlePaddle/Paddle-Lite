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

#include "lite/kernels/arm/affine_grid_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void AffineGridCompute::PrepareForRun() {
  auto& param = Param<operators::AffineGridParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  const lite::Tensor* x = param.X;
  const float* din = x->data<float>();
  lite::Tensor* out = param.Out;
  float* dout = param.Out->mutable_data<float>();
  int H = out->dims()[1];
  int W = out->dims()[2];
  if (param.output_shape.size() == 0) {
    const auto out_shape = param.OutputShape->data<int>();
    H = out_shape[2];
    W = out_shape[3];
  } else {
    H = param.output_shape[2];
    W = param.output_shape[3];
  }
  bool align_corners = param.align_corners;
  std::vector<float> vvh(H);
  vh = vvh.data();
  std::vector<float> vvw(W);
  vw = vvw.data();
  int out_size = H * W * 3;
  vhw3.resize(out_size);
  hw3 = vhw3.data();
  float scale = 2 / (static_cast<float>(H) - 1);
  float start = -1.0f;
  if (!align_corners) {
    scale = 2 / static_cast<float>(H);
    start *= (static_cast<float>(H) - 1) / static_cast<float>(H);
  }
  for (int i = 0; i < H; i++) {
    vh[i] = start + scale * i;
  }
  scale = 2 / (static_cast<float>(W) - 1);
  start = -1.0f;
  if (!align_corners) {
    scale = 2 / static_cast<float>(W);
    start *= (static_cast<float>(W) - 1) / static_cast<float>(W);
  }
  for (int i = 0; i < W; i++) {
    vw[i] = start + i * scale;
  }

  for (int i = 0; i < out_size; i += 3) {
    hw3[i] = 1;
    hw3[i + 1] = 1;
    hw3[i + 2] = 1;
  }

  for (int i = 0; i < H * W; i++) {
    hw3[i * 3 + 1] = vh[i / W];
  }
  for (int i = 0; i < H * W; i++) {
    hw3[i * 3] = vw[i % W];
  }
  return;
}
void AffineGridCompute::Run() {
  auto& param = Param<operators::AffineGridParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  const lite::Tensor* x = param.X;
  lite::Tensor* out = param.Out;
  int N = x->dims()[0];
  int H = out->dims()[1];
  int W = out->dims()[2];
  const float* din = x->data<float>();
  float* dout = param.Out->mutable_data<float>();
  operators::ActivationParam act_param;
  act_param.has_active = false;
  for (int i = 0; i < N; i++) {
    lite::arm::math::sgemm(false,
                           true,
                           H * W,
                           2,
                           3,
                           1.f,
                           hw3,
                           3,
                           din,
                           3,
                           0.f,
                           dout,
                           2,
                           nullptr,
                           false,
                           act_param,
                           &ctx);

    din += 6;
    dout += H * W * 2;
  }

  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(affine_grid,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::AffineGridCompute,
                     def)
    .BindInput("Theta", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OutputShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
