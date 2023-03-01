// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/conv2d_transpose_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename TGEMM,
          typename TW,
          typename TX,
          typename TY,
          PrecisionType PType>
void Conv2dTransposeCompute<TGEMM, TW, TX, TY, PType>::PrepareForRun() {
  int cur_dev_idx = 0;

  XPU_CALL(xpu_current_device(&cur_dev_idx));
  XPU_CALL(xpu_device_get_attr(&cur_dev_attr_, XPUATTR_MODEL, cur_dev_idx));
  if (cur_dev_attr_ <= 1) {
    VLOG(4) << "Currents XPU device : XPU1";
  } else if (cur_dev_attr_ >= 2 && cur_dev_attr_ <= 299) {
    VLOG(4) << "Currents XPU device : XPU2";
  } else if (cur_dev_attr_ >= 300 && cur_dev_attr_ <= 599) {
    VLOG(4) << "Currents XPU device : XPU3";
  } else {
    VLOG(4) << "invaid XPU device";
  }

  // do filter quant for xpu2/xpu3
  if (cur_dev_attr_ > 1) {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<XPUContext>();
    int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
    auto filter_ptr = param.filter->template data<float>();
    auto filter_dims = param.filter->dims();
    auto filter_size = param.filter->numel();

    std::vector<float> cpu_weights(filter_size);
    lite::TargetWrapperXPU::MemcpySync(cpu_weights.data(),
                                       filter_ptr,
                                       sizeof(float) * filter_size,
                                       IoDirection::DtoH);
    // data precision cast FP16 <-> FP32
    xpu_quant_filter_ =
        TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, TW>(
            cpu_weights.data(), filter_dims, false, max_ptr_size);

    if (param.activation_param.has_active) {
      if (param.fuse_relu) {
        act_ = xdnn::Activation_t::RELU;
      } else if (param.fuse_sigmoid) {
        act_ = xdnn::Activation_t::SIGMOID;
      } else if (param.fuse_tanh) {
        act_ = xdnn::Activation_t::TANH;
      } else {
        act_ = xdnn::Activation_t::LINEAR;
      }
    }
  }
}

template <typename TGEMM,
          typename TW,
          typename TX,
          typename TY,
          PrecisionType PType>
void Conv2dTransposeCompute<TGEMM, TW, TX, TY, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& out_dims = param.output->dims();
  auto& w_dims = param.filter->dims();
  auto& in_dims = param.x->dims();

  int groups = param.groups;
  auto& strides = param.strides;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  if (param.output_padding.empty()) {
    if (cur_dev_attr_ <= 1) {
      int ret = xdnn::conv2d_transpose<float, float, float, TGEMM>(
          ctx.GetRawContext(),
          param.x->template data<float>(),
          param.filter->template data<float>(),
          param.output->template mutable_data<float>(TARGET(kXPU)),
          in_dims[0],
          in_dims[1],
          in_dims[2],
          in_dims[3],
          out_dims[1],
          std::vector<int>{static_cast<int>(w_dims[2]),
                           static_cast<int>(w_dims[3])},
          strides,
          paddings,
          dilations,
          groups,
          nullptr,
          nullptr,
          nullptr,
          true);
      CHECK_EQ(ret, 0);
    } else {
      const auto* bias = (param.bias != nullptr)
                             ? param.bias->template data<float>()
                             : nullptr;
      if (param.output_size.size()) {
        int ret = xdnn::conv2d_transpose_fusion_v2<TX, TW, TY, int16_t>(
            ctx.GetRawContext(),
            param.x->template data<TX>(),
            reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_),
            param.output->template mutable_data<TY>(TARGET(kXPU)),
            in_dims[0],
            in_dims[1],
            out_dims[2],
            out_dims[3],
            out_dims[1],
            std::vector<int>{static_cast<int>(w_dims[2]),
                             static_cast<int>(w_dims[3])},
            strides,
            paddings,
            dilations,
            groups,
            nullptr,
            reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
            nullptr,
            bias,
            act_,
            true);
        CHECK_EQ(ret, 0);
      } else {
        int ret = xdnn::conv2d_transpose_fusion<TX, TW, TY, int16_t>(
            ctx.GetRawContext(),
            param.x->template data<TX>(),
            reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_),
            param.output->template mutable_data<TY>(TARGET(kXPU)),
            in_dims[0],
            in_dims[1],
            in_dims[2],
            in_dims[3],
            out_dims[1],
            std::vector<int>{static_cast<int>(w_dims[2]),
                             static_cast<int>(w_dims[3])},
            strides,
            paddings,
            dilations,
            groups,
            nullptr,
            reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
            nullptr,
            bias,
            act_,
            true);
        CHECK_EQ(ret, 0);
      }
    }
  } else {
    int n = in_dims[0];
    int yc = in_dims[1];
    int yh = in_dims[2];
    int yw = in_dims[3];
    int xc = out_dims[1];
    int xh = out_dims[2];
    int xw = out_dims[3];
    int kh = w_dims[2];
    int kw = w_dims[3];

    XPUScratchPadGuard x_trans_guard =
        TargetWrapperXPU::MallocScratchPad((param.x->numel()) * sizeof(TX));
    TX* x_trans = reinterpret_cast<TX*>(x_trans_guard->addr_);
    XPUScratchPadGuard x_col_before_concat_guard =
        TargetWrapperXPU::MallocScratchPad((n * yh * yw * kh * kw * xc) *
                                           sizeof(TX));
    TX* x_col_before_concat =
        reinterpret_cast<TX*>(x_col_before_concat_guard->addr_);
    XPUScratchPadGuard x_col_guard = TargetWrapperXPU::MallocScratchPad(
        (n * yh * yw * kh * kw * xc) * sizeof(TX));
    TX* x_col = reinterpret_cast<TX*>(x_col_guard->addr_);

    const float* weight = param.filter->template data<float>();
    int ret = xdnn::transpose<TX>(ctx.GetRawContext(),
                                  param.x->template data<TX>(),
                                  x_trans,
                                  {n, groups, yc / groups, yh, yw},
                                  {1, 0, 3, 4, 2});
    CHECK_EQ(ret, 0);
    for (int g = 0; g < groups; g++) {
      const TX* curr_y = x_trans + g * n * yh * yw * (yc / groups);
      const float* curr_w =
          weight + g * (yc / groups) * (xc / groups) * kh * kw;
      TX* curr_x =
          x_col_before_concat + g * n * yh * yw * (xc / groups) * kh * kw;
      int mac_m = n * yh * yw;
      int mac_k = yc / groups;
      int mac_n = xc / groups * kh * kw;
      ret = xdnn::fc<TX, float, TX, TGEMM>(ctx.GetRawContext(),
                                           curr_y,
                                           curr_w,
                                           curr_x,
                                           mac_m,
                                           mac_n,
                                           mac_k,
                                           false,
                                           false,
                                           nullptr,
                                           nullptr,
                                           nullptr);
      CHECK_EQ(ret, 0);
    }
    ret = xdnn::transpose<TX>(ctx.GetRawContext(),
                              x_col_before_concat,
                              x_col,
                              {groups, n * yh * yw, (xc / groups) * kh * kw},
                              {1, 0, 2});
    CHECK_EQ(ret, 0);

    ret =
        xdnn::col2im<TY>(ctx.GetRawContext(),
                         x_col,
                         param.output->template mutable_data<TY>(TARGET(kXPU)),
                         n,
                         xc,
                         xh,
                         xw,
                         std::vector<int>{static_cast<int>(w_dims[2]),
                                          static_cast<int>(w_dims[3])},
                         strides,
                         paddings,
                         dilations,
                         true);
    CHECK_EQ(ret, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using Conv2dTransposeFp32 = xpu::
    Conv2dTransposeCompute<int16_t, float, float, float, PRECISION(kFloat)>;
using Conv2dTransposeFp16 = xpu::Conv2dTransposeCompute<int16_t,
                                                        int16_t,
                                                        float16,
                                                        float16,
                                                        PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    conv2d_transpose, kXPU, kFloat, kNCHW, Conv2dTransposeFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(conv2d_transpose,
                     kXPU,
                     kFP16,
                     kNCHW,
                     Conv2dTransposeFp16,
                     DISABLE_XPU1_fp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
