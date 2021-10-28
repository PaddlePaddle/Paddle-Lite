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
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUFcCompute::PrepareForRun() {
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<param_t>();
  auto w_ptr = param.w->data<float>();
  auto w_len = param.w->numel();
  auto weight_dims = param.w->dims();
  bool quant_int8 = false;
  if (param.quant_w_max > 0.f) {
    quant_int8 = true;
  }
  // max
  if (!quant_int8) {
    w_max = paddle::lite::xpu::math::FindMaxAbs(w_ptr, w_len);
    int max_ptr_size = get_max_ptr_size(ctx.GetRawContext());
    std::vector<float> w_max_v(max_ptr_size, w_max);
    weight_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(weight_max_guard_->addr_),
                        w_max_v.data(),
                        max_ptr_size * sizeof(float),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    input_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
  }
  // transpose
  if (quant_int8) {
    std::vector<int8_t> transpose_w_int8(w_len, 0);
    paddle::lite::xpu::math::Transpose<int8_t>(
        reinterpret_cast<const int8_t*>(w_ptr),
        transpose_w_int8.data(),
        weight_dims[0],
        weight_dims[1]);
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(int8_t));
    XPU_CALL(xpu_memcpy(reinterpret_cast<int8_t*>(quant_weight_guard_->addr_),
                        transpose_w_int8.data(),
                        w_len * sizeof(int8_t),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    return;
  }
  std::vector<float> transpose_w(w_len, 0);
  paddle::lite::xpu::math::Transpose(
      w_ptr, transpose_w.data(), weight_dims[0], weight_dims[1]);
  // quant
  if (param.precision == "int31") {
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(float));
    XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(quant_weight_guard_->addr_),
                        transpose_w.data(),
                        w_len * sizeof(float),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  } else if (param.precision == "int16") {
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(int16_t));
    std::vector<int16_t> quant_weight_cpu(w_len, 0);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        transpose_w.data(), quant_weight_cpu.data(), w_max, w_len);
    XPU_CALL(xpu_memcpy(reinterpret_cast<int16_t*>(quant_weight_guard_->addr_),
                        quant_weight_cpu.data(),
                        w_len * sizeof(int16_t),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  } else if (param.precision == "int8") {
    quant_weight_guard_ =
        TargetWrapperXPU::MallocScratchPad(w_len * sizeof(int8_t));
    std::vector<int8_t> quant_weight_cpu(w_len, 0);
    paddle::lite::xpu::math::ConvertFP32ToInt8(
        transpose_w.data(), quant_weight_cpu.data(), w_max, w_len);
    XPU_CALL(xpu_memcpy(reinterpret_cast<int8_t*>(quant_weight_guard_->addr_),
                        quant_weight_cpu.data(),
                        w_len * sizeof(int8_t),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  }
}

void XPUFcCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto input_dims = param.input->dims();
  auto in_mat_dims = input_dims.Flatten2D(param.in_num_col_dims);
  int m = in_mat_dims[0];
  int k = in_mat_dims[1];
  int n = param.w->dims()[1];
  bool quant_int8 = param.quant_w_max > 0.f;

  float* output_max = quant_int8
                          ? nullptr
                          : param.output_max->mutable_data<float>(TARGET(kXPU));
  const auto* bias = param.has_bias ? param.bias->data<float>() : nullptr;
  const float* input_max =
      quant_int8 ? nullptr
                 : (param.input_max ? param.input_max->data<float>() : nullptr);
  xdnn::Activation_t act((xdnn::Activation_t::act_enum)param.act_type);
  if (param.act_type == 5) {
    act.leaky_alpha = param.act_param;
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (param.act_type == 15) {
    act.hard_sigmoid_slope = param.act_param;
  }
  // TODO(weihaoji): remove fc_int31 and fc_int16 after xpu fc wrapper refactor
  if (param.precision == "int31") {
    int r = xdnn::fc_int31(
        ctx.GetRawContext(),        /* context */
        false,                      /* TransA */
        true,                       /* TransB */
        m,                          /* m */
        n,                          /* n */
        k,                          /* k */
        1.0f,                       /* alpha */
        param.input->data<float>(), /* A */
        nullptr,                    /* max_a ptr */
        reinterpret_cast<const float*>(quant_weight_guard_->addr_), /* B */
        w_max,                                                      /* max_b */
        0.0f,                                                       /* beta */
        param.output->mutable_data<float>(TARGET(kXPU)),            /* C */
        nullptr, /* max_c ptr */
        bias,    /* bias */
        act /* act_type */);
    CHECK_EQ(r, 0);
    r = xdnn::findmax<float>(
        ctx.GetRawContext(), param.output->data<float>(), m * n, output_max);
    CHECK_EQ(r, 0);
  } else if (param.precision == "int16") {
    int r = 0;
    if (input_max == nullptr) {
      r = xdnn::findmax<float>(
          ctx.GetRawContext(),
          param.input->data<float>(),
          m * k,
          reinterpret_cast<float*>(input_max_guard_->addr_));
      CHECK_EQ(r, 0);
    }
    r = xdnn::fc_fusion<float, int16_t, float, int16_t>(
        ctx.GetRawContext(),                                           // ctx
        param.input->data<float>(),                                    // x
        reinterpret_cast<const int16_t*>(quant_weight_guard_->addr_),  // w
        param.output->mutable_data<float>(TARGET(kXPU)),               // y
        m,                                                             // m
        n,                                                             // n
        k,                                                             // k
        false,  // x_trans
        true,   // w_trans
        (input_max == nullptr)
            ? reinterpret_cast<const float*>(input_max_guard_->addr_)
            : input_max,                                           // x_maxptr
        reinterpret_cast<const float*>(weight_max_guard_->addr_),  // w_maxptr
        output_max,                                                // y_maxptr
        k,                                                         // ldx
        k,                                                         // ldw
        n,                                                         // ldy
        1.0f,                                                      // alpha
        0.0f,                                                      // beta
        bias,                                                      // bias
        act);                                                      // act

    CHECK_EQ(r, 0);
  } else if (param.precision == "int8") {
    bool x_trans = false;
    bool w_trans = true;
    if (quant_int8) {
      int r = xdnn::fc_int8(
          ctx.GetRawContext(),
          false,
          true,
          m,
          n,
          k,
          1.0f,
          param.input->data<float>(),
          param.quant_input_max,
          reinterpret_cast<const int8_t*>(quant_weight_guard_->addr_),
          param.quant_w_max,
          0.f,
          param.output->mutable_data<float>(TARGET(kXPU)),
          bias,
          act);
      CHECK_EQ(r, 0);
      return;
    }
    int ldx = (x_trans ? m : k);
    int ldw = (w_trans ? k : n);
    int ldy = n;
    int r = xdnn::fc_fusion<float, int8_t, float, int8_t>(
        ctx.GetRawContext(),        /* context */
        param.input->data<float>(), /* x */
        reinterpret_cast<const int8_t*>(quant_weight_guard_->addr_),
        param.output->mutable_data<float>(TARGET(kXPU)),    /* y */
        m,                                                  /* m */
        n,                                                  /* n */
        k,                                                  /* k */
        x_trans,                                            /* x_trans */
        w_trans,                                            /* w_trans */
        input_max,                                          /* x_max */
        reinterpret_cast<float*>(weight_max_guard_->addr_), /* w_max */
        output_max,                                         /* y_max */
        ldx,                                                /* ldx */
        ldw,                                                /* ldw */
        ldy,                                                /* ldy */
        1.0f,                                               /* alpha */
        0.0f,                                               /* beta */
        bias,                                               /* bias */
        act);                                               /* act_type */
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "Unsupport XPUFC Precision: " << param.precision;
  }
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
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
