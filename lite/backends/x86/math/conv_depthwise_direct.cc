/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/conv_depthwise_direct.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void conv_depthwise_direct(const float* din,
                        float* dout,
                        int num,
                        int ch_out,
                        int h_out,
                        int w_out,
                        int ch_in,
                        int h_in,
                        int w_in,
                        const float* weights,
                        const float* bias,
                        const operators::ConvParam& param) {
                            auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int pad_w = paddings[2];
  int stride = param.strides[1];
  int pad = pad_w;
  bool flag_bias = param.bias != nullptr;
  int kernel_h = param.filter->dims()[2];

  if (kernel_h == 3) {
    if (stride == 1) {
      if (pad == 1) { //TODO pad = 0
        conv_depthwise_3x3s1_p1_direct(din,
                                        dout,
                                        num,
                                        ch_out,
                                        h_out,
                                        w_out,
                                        ch_in,
                                        h_in,
                                        w_in,
                                        weights,
                                        bias,
                                        pad,
                                        flag_bias,
                                        act_param);
      }
    } else if (stride == 2) {
      if (pad == 1) {//TODO pad = 0
        conv_depthwise_3x3s2_p1_direct(din,
                                        dout,
                                        num,
                                        ch_out,
                                        h_out,
                                        w_out,
                                        ch_in,
                                        h_in,
                                        w_in,
                                        weights,
                                        bias,
                                        pad,
                                        flag_bias,
                                        act_param);
      }
    }
  }     


}
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddl    