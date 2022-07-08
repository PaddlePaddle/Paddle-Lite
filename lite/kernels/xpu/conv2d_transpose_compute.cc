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
          typename DX,
          typename DY,
          PrecisionType PType>
void Conv2dTransposeCompute<TGEMM, TW, DX, DY, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto filter_ptr = param.filter->template data<float>();
  auto filter_dims = param.filter->dims();
  xpu_quant_filter_ =
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, TW>(
          filter_ptr, filter_dims, false);

#ifdef LITE_WITH_XPU
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
#endif
}

template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
void Conv2dTransposeCompute<TGEMM, TW, DX, DY, PType>::Run() {
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
      // conv2d_transpose_fusion only support kl2，conv2d_transpose only support
      // data precision FP32
      CHECK_EQ(sizeof(DX), sizeof(float));
      int ret = xdnn::conv2d_transpose<float, float, float, int16_t>(
          ctx.GetRawContext(),
          param.x->template data<float>(),
          reinterpret_cast<const float*>(
              xpu_quant_filter_.data_ptr_), /* weight */
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
          reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
          nullptr,
          true);
      CHECK_EQ(ret, 0);
    } else {
      int ret = xdnn::conv2d_transpose_fusion<DX, TW, DY, TGEMM>(
          ctx.GetRawContext(),
          param.x->template data<DX>(),
          reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_), /* weight */
          param.output->template mutable_data<DY>(TARGET(kXPU)),
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
          nullptr,
          xdnn::Activation_t::LINEAR,
          true);
      CHECK_EQ(ret, 0);
    }

  } else {
    CHECK_EQ(sizeof(DX), sizeof(DY));

    int n = in_dims[0];
    int yc = in_dims[1];
    int yh = in_dims[2];
    int yw = in_dims[3];
    int xc = out_dims[1];
    int xh = out_dims[2];
    int xw = out_dims[3];
    int kh = w_dims[2];
    int kw = w_dims[3];
    DX* x_trans = nullptr;
    XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&x_trans),
                        (param.x->numel()) * sizeof(DX)));
    DX* x_col_before_concat = nullptr;
    XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&x_col_before_concat),
                        (n * yh * yw * kh * kw * xc) * sizeof(DX)));
    DX* x_col = nullptr;
    XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&x_col),
                        (n * yh * yw * kh * kw * xc) * sizeof(DX)));
    const TW* weight = reinterpret_cast<const TW*>(xpu_quant_filter_.data_ptr_);
    int ret = xdnn::transpose<DX>(ctx.GetRawContext(),
                                  param.x->template data<DX>(),
                                  x_trans,
                                  {n, groups, yc / groups, yh, yw},
                                  {1, 0, 3, 4, 2});
    CHECK_EQ(ret, 0);

    for (int g = 0; g < groups; g++) {
      const DX* curr_y = x_trans + g * n * yh * yw * (yc / groups);
      const TW* curr_w = weight + g * (yc / groups) * (xc / groups) * kh * kw;
      DX* curr_x =
          x_col_before_concat + g * n * yh * yw * (xc / groups) * kh * kw;
      int mac_m = n * yh * yw;
      int mac_k = yc / groups;
      int mac_n = xc / groups * kh * kw;
      ret = xdnn::fc<DX, TW, DY, TGEMM>(
          ctx.GetRawContext(),
          curr_y,
          reinterpret_cast<const TW*>(curr_w),
          curr_x,
          mac_m,
          mac_n,
          mac_k,
          false,
          false,
          nullptr,
          reinterpret_cast<const float*>(xpu_quant_filter_.max_ptr_),
          nullptr);
      CHECK_EQ(ret, 0);
    }
    ret = xdnn::transpose<DX>(ctx.GetRawContext(),
                              x_col_before_concat,
                              x_col,
                              {groups, n * yh * yw, (xc / groups) * kh * kw},
                              {1, 0, 2});
    CHECK_EQ(ret, 0);

    ret =
        xdnn::col2im<DY>(ctx.GetRawContext(),
                         x_col,
                         param.output->template mutable_data<DY>(TARGET(kXPU)),
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
    XPU_CALL(xpu_free(x_trans));
    XPU_CALL(xpu_free(x_col_before_concat));
    XPU_CALL(xpu_free(x_col));
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using Conv2dTranspose_FP16_FP32_FP32 = xpu::
    Conv2dTransposeCompute<int16_t, int16_t, float, float, PRECISION(kFloat)>;

using Conv2dTransposeFp16 = xpu::Conv2dTransposeCompute<int16_t,
                                                        int16_t,
                                                        float16,
                                                        float16,
                                                        PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    conv2d_transpose, kXPU, kFloat, kNCHW, Conv2dTranspose_FP16_FP32_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(conv2d_transpose,
                     kXPU,
                     kFP16,
                     kNCHW,
                     Conv2dTransposeFp16,
                     DISABLE_XPU1_Conv2dTransposeFp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
