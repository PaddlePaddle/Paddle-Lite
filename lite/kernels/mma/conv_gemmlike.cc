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

#include "lite/kernels/mma/conv_gemmlike.h"
#include <vector>
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/backends/mma/lldrv/mmadrv.h"
#include "lite/backends/mma/lldrv/utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mma {

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

  //------------------------------------------------------------------------------------
  if (kh > 1 && kw > 1) {
    int i, j, il, kl, ol, l, m, n, k;
    lite::mma::mma_conv_s conv;

    conv.at = static_cast<uint32_t>(param.activation_param.active_type);
    if (conv.at == 4) {
      alpha = param.activation_param.Leaky_relu_alpha;
    }
    conv.ng = param.groups;

    conv.i.in = i_dims[0];
    conv.i.ic = i_dims[1];
    conv.i.ih = i_dims[2];
    conv.i.iw = i_dims[3];
    conv.i.pl = paddings[2];  // left
    conv.i.pr = paddings[3];  // right
    conv.i.pt = paddings[0];  // top
    conv.i.pb = paddings[1];  // bottom
    conv.i.dy = dilations[0];
    conv.i.dx = dilations[1];

    conv.k.kh = w_dims[2];
    conv.k.kw = w_dims[3];
    conv.k.hs = param.strides[0];
    conv.k.ws = param.strides[1];

    conv.o.on = o_dims[0];
    conv.o.oc = o_dims[1];
    conv.o.oh = o_dims[2];
    conv.o.ow = o_dims[3];

    il = conv.i.in * conv.i.ic * conv.i.ih * conv.i.iw;
    kl = conv.o.oc * conv.i.ic * conv.k.kh * conv.k.kw;
    ol = conv.o.on * conv.o.oc * conv.o.oh * conv.o.ow;
    conv.ia = static_cast<int8_t*>(lite::mma::mma_minput(il * sizeof(int8_t)));
    conv.ka = static_cast<int8_t*>(lite::mma::mma_mkernel(kl * sizeof(int8_t)));
    conv.oa =
        static_cast<int32_t*>(lite::mma::mma_moutput(ol * sizeof(int32_t)));
    if (conv.ia && conv.ka && conv.oa) {
      float fd = lite::mma::find_max(i_data, il);
      float fw = lite::mma::find_max(w_data, kl);

      fd = 127.0 / fd;
      fw = 127.0 / fw;

      // y = 127.0 / fmax
      // y = x * scale;
      lite::mma::quantize_s8(i_data, conv.ia, il, fd);
      lite::mma::quantize_s8(w_data, conv.ka, kl, fw);

      // perform conv2d
      if (lite::mma::mma_conv(&conv)) {
        std::cout << "mma_conv error" << std::endl;
      }
      // Convert int32 back to fp32, [n,c,h,w]
      // 1. y = x / scale
      // 2. y = x + b
      // 3. y = f(x)
      int hw = conv.o.oh * conv.o.ow;
      for (i = 0; i < conv.o.on; i++) {
        for (j = 0; j < conv.o.oc; j++) {
          m = i * conv.o.oc + j;
          n = m * hw;
          for (l = 0; l < hw; l++) {
            k = n + l;
            o_data[k] = static_cast<float>(conv.oa[k] / fd / fw);
            if (b_data) o_data[k] += b_data[j];
            if (conv.at == 1) {  // relu
              o_data[k] = o_data[k] > 0.0 ? o_data[k] : 0.0;
            } else if (conv.at == 2) {  // relu6
              o_data[k] = o_data[k] > 0.0 ? o_data[k] : 0.0;
              o_data[k] = o_data[k] > 6.0 ? 6.0 : o_data[k];
            } else if (conv.at == 4) {  // leakyRelu
              if (o_data[k] < 0.0) o_data[k] = o_data[k] * alpha;
            }
          }
        }
      }
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

}  // namespace mma
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
