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

#include "lite/kernels/xpu/search_fc_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SearchFcCompute::PrepareForRun() {
  maxs_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(XPU_MAX_LOD_SIZE * sizeof(float));
}

void SearchFcCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* bottom = param.X;
  auto* w = param.W;
  auto* b = param.b;
  auto* top = param.Out;
  float w_max = param.__xpu__w_max;
  int out_size = param.out_size;
  bool fuse_relu = param.fuse_relu;
  bool float_to_fix = param.__xpu__float_to_fix;
  CHECK(float_to_fix) << "W should be fixed point";

  int batch = bottom->dims()[0];
  int _out = w->dims()[0];
  int _in = w->dims()[1];

  xdnn::Activation_t act = xdnn::Activation_t::LINEAR;
  if (fuse_relu) {
    act = xdnn::Activation_t::RELU;
  }

  std::vector<int64_t> top_dims{bottom->dims()[0], out_size};
  top->Resize(top_dims);

  const auto* bottom_data = bottom->data<float>();
  const auto* weights = w->data<int16_t>();
  const auto* bias_data = b->data<float>();
  auto* top_data = top->mutable_data<float>(TARGET(kXPU));

  float* maxs_xpu = reinterpret_cast<float*>(maxs_xpu_guard_->addr_);
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  std::vector<float> maxs_cpu(2 * max_ptr_size, 0.0f);
  maxs_cpu[max_ptr_size] = w_max;
  XPU_CALL(xpu_memcpy(maxs_xpu,
                      &maxs_cpu[0],
                      2 * max_ptr_size * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int r = xdnn::findmax<float>(
      ctx.GetRawContext(), bottom_data, maxs_xpu, batch * _in);
  CHECK_EQ(r, 0);
  r = xdnn::fc_fusion<float, int16_t, float, int16_t>(
      ctx.GetRawContext(),     /* ctx */
      bottom_data,             /* data_a */
      weights,                 /* data_b */
      top_data,                /* data_c */
      batch,                   /* m */
      _out,                    /* n */
      _in,                     /* k */
      false,                   /* trans_a */
      true,                    /* trans_b */
      maxs_xpu,                /* max_a */
      maxs_xpu + max_ptr_size, /* max_b */
      nullptr,                 /* max_c */
      _in,                     /* lda */
      _in,                     /* ldb */
      _out,                    /* ldc */
      1.0f,                    /* alpha */
      0.0f,                    /* beta */
      bias_data,               /* bias */
      act /* act */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_fc,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SearchFcCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
