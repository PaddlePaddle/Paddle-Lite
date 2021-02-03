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

#include "lite/kernels/xpu/__xpu__fc_compute.h"
#include "lite/backends/xpu/target_wrapper.h"  // XPUScratchPadGuard
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
void XPUFcCompute::PrepareForRun() {
  XPUMalloc(&x_max_);
  XPUMalloc(&w_max_);
  XPUMalloc(&y_max_);

  auto& param = this->Param<param_t>();
  float w_maxs[4] = {param.w_max, 0.0f, 0.0f, 0.0f};
  XPU_CALL(xpu_memcpy(
      w_max_, w_maxs, 4 * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void XPUFcCompute::XPUMalloc(float** max_ptr) {
  XPUScratchPadGuard max_guard =
      TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
  *max_ptr = reinterpret_cast<float*>(max_guard->addr_);
}

void XPUFcCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto input_dims = param.input->dims();
  param.in_mat_dims = input_dims.Flatten2D(param.in_num_col_dims);
  int m = param.in_mat_dims[0];
  int k = param.in_mat_dims[1];
  int n = param.w->dims()[1];
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;
  xdnn::Activation_t act_type = (param.activation_type == "relu")
                                    ? xdnn::Activation_t::RELU
                                    : xdnn::Activation_t::LINEAR;

  int r = -1;
  if (param.precision == "int31") {
    r = xdnn::fc_int31(ctx.GetRawContext(),        /* context */
                       false,                      /* TransA */
                       param.transpose_w,          /* TransB */
                       m,                          /* m */
                       n,                          /* n */
                       k,                          /* k */
                       1.0f,                       /* alpha */
                       param.input->data<float>(), /* A */
                       nullptr,                    /* max_a ptr */
                       param.w->data<float>(),     /* B */
                       param.w_max,                /* max_b */
                       0.0f,                       /* beta */
                       param.output->mutable_data<float>(TARGET(kXPU)), /* C */
                       nullptr, /* max_c ptr */
                       bias,    /* bias */
                       act_type /* act_type */);
  } else if (param.precision == "int8") {
    // x_max
    r = xdnn::findmax<float>(
        ctx.GetRawContext(), param.input->data<float>(), m * k, x_max_);
    CHECK_EQ(r, 0);

    bool x_trans = false;
    bool w_trans = param.transpose_w;
    int ldx = (x_trans ? m : k);
    int ldw = (w_trans ? k : n);
    int ldy = n;
    r = xdnn::fc_fusion<float, int8_t, float, int8_t>(
        ctx.GetRawContext(),                                     /* context */
        param.input->data<float>(),                              /* x */
        reinterpret_cast<const int8_t*>(param.w->data<float>()), /* w */
        param.output->mutable_data<float>(TARGET(kXPU)),         /* y */
        m,                                                       /* m */
        n,                                                       /* n */
        k,                                                       /* k */
        false,                                                   /* x_trans */
        param.transpose_w,                                       /* w_trans */
        x_max_,                                                  /* x_max */
        w_max_,                                                  /* w_max */
        y_max_,                                                  /* y_max */
        ldx,                                                     /* ldx */
        ldw,                                                     /* ldw */
        ldy,                                                     /* ldy */
        1.0f,                                                    /* alpha */
        0.0f,                                                    /* beta */
        bias,                                                    /* bias */
        act_type /* act_type */);
  } else {
    r = xdnn::fc_int16(
        ctx.GetRawContext(),                                      /* context */
        false,                                                    /* TransA */
        param.transpose_w,                                        /* TransB */
        m,                                                        /* m */
        n,                                                        /* n */
        k,                                                        /* k */
        1.0f,                                                     /* alpha */
        param.input->data<float>(),                               /* A */
        reinterpret_cast<const int16_t*>(param.w->data<float>()), /* B */
        param.w_max,                                              /* max_b */
        0.0f,                                                     /* beta */
        param.output->mutable_data<float>(TARGET(kXPU)),          /* C */
        bias,                                                     /* bias */
        act_type /* act_type */);
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__fc,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUFcCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
