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

#include "lite/kernels/intel_fpga/conv_gemmlike.h"
#include <vector>
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/utils/log/logging.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<float>();
  if (flag_trans_weights_) {
    weights = weights_.data<float>();
  }
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto i_data = param.x->data<float>();
  auto w_data = param.filter->data<float>();
  auto o_data = param.output->mutable_data<float>();
  auto i_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  int iw, ih, ic, bs, ow, oh, oc;
  float alpha;

  iw = i_dims[3];  // nchw
  ih = i_dims[2];
  ic = i_dims[1];
  bs = i_dims[0];
  oh = o_dims[2];
  ow = o_dims[3];
  oc = o_dims[1];

  int kh = w_dims[2];
  int kw = w_dims[3];

  if (kh > 1 && kw > 1) {
    int i, j, il, kl, ol, l, m, n, k;
    intelfpga_conv2d_s conv;

    conv.at = static_cast<uint32_t>(param.activation_param.active_type);
    conv.ng = param.groups;
    switch (conv.at) {
      case 1:
        conv.at = INTELFPGA_ACT_RELU;
        break;
      case 2:
        conv.at = INTELFPGA_ACT_RELU6;
        break;
      case 4:
        conv.at = INTELFPGA_ACT_LEAKYRELU;
        conv.alpha = param.activation_param.Leaky_relu_alpha;
        break;
      default:
        conv.at = INTELFPGA_ACT_NONE;
        break;
    }
    conv.ia = const_cast<float*>(i_data);
    conv.ka = const_cast<float*>(w_data);
    conv.ba = const_cast<float*>(b_data);
    conv.oa = const_cast<float*>(o_data);
    conv.ip.in = i_dims[0];
    conv.ip.ic = i_dims[1];
    conv.ip.ih = i_dims[2];
    conv.ip.iw = i_dims[3];
    conv.ip.pl = paddings[2];  // left
    conv.ip.pr = paddings[3];  // right
    conv.ip.pt = paddings[0];  // top
    conv.ip.pb = paddings[1];  // bottom
    conv.ip.dy = dilations[0];
    conv.ip.dx = dilations[1];

    conv.kp.kh = w_dims[2];
    conv.kp.kw = w_dims[3];
    conv.kp.hs = param.strides[0];
    conv.kp.ws = param.strides[1];

    conv.op.on = o_dims[0];
    conv.op.oc = o_dims[1];
    conv.op.oh = o_dims[2];
    conv.op.ow = o_dims[3];
    if (intelfpga_conv2d(&conv)) {
      LOG(WARNING) << "[IntelFPGA] Conv_Compute failed";
    }
  } else {
    if (flag_1x1gemm_) {
      lite::arm::math::conv1x1s1_gemm(i_data,
                                      o_data,
                                      bs,
                                      oc,
                                      oh,
                                      ow,
                                      ic,
                                      ih,
                                      iw,
                                      weights,
                                      b_data,
                                      param,
                                      &ctx);
    } else {
      lite::arm::math::conv_im2col_gemm(i_data,
                                        o_data,
                                        bs,
                                        oc,
                                        oh,
                                        ow,
                                        ic,
                                        ih,
                                        iw,
                                        weights,
                                        b_data,
                                        param,
                                        &ctx);
    }
  }
}

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
