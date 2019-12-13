/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef BILINEAR_INTERP_OP

#include "operators/kernel/bilinear_interp_kernel.h"
// #include "operators/kernel/central-arm-func/bilinear_interp_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void BilinearInterpCompute(const BilinearInterpParam<FPGA>& param) {

  zynqmp::Tensor* input_x = param.InputX()->zynqmpTensor();
  auto out_dims = param.Out()->dims();
  // auto* input = param.InputX()->data<float>();
  auto out_size_t = param.InputOutPutSize();

  int out_h = param.OutH();
  int out_w = param.OutW();
  if (out_size_t != nullptr) {
    auto out_size_data = out_size_t->data<int>();
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }

  zynqmp::Tensor input_float;
  // input_float.setDataLocation(CPU);
  // input_x.setAligned(true);
  input_x->unalignImage();
  float* input = input_float.mutableData<float>(zynqmp::FP32, input_x->shape());
  input_float.copyFrom(input_x);

  auto* output = param.Out()->mutable_data<float>(
      {out_dims[0], out_dims[1], out_h, out_w});
  auto batch_size = param.InputX()->dims()[0];
  auto channels = param.InputX()->dims()[1];
  auto in_h = param.InputX()->dims()[2];
  auto in_w = param.InputX()->dims()[3];

  auto in_hw = in_h * in_w;
  auto out_hw = out_h * out_w;
  auto in_chw = channels * in_hw;
  auto out_chw = channels * out_hw;

  float ratio_h =
      (out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : 0.f;
  float ratio_w =
      (out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : 0.f;

  if (in_h == out_h && in_w == out_w) {
    memcpy(output, input, param.InputX()->numel() * sizeof(float));
  } else {
    for (int k = 0; k < batch_size; ++k) {  // loop for batches
      for (int i = 0; i < out_h; ++i) {     // loop for images
        int h = ratio_h * i;
        int hid = (h < in_h - 1) ? in_w * channels : 0;
        float h1lambda = ratio_h * i - h;
        float h2lambda = 1.f - h1lambda;

        for (int j = 0; j < out_w; ++j) {
          int w = ratio_w * j;
          int wid = (w < in_w - 1) ? channels : 0;
          float w1lambda = ratio_w * j - w;
          float w2lambda = 1.f - w1lambda;
          // calculate four position for bilinear interpolation
          const float* in_pos = &input[(k * in_h * in_w + h * in_w + w) * channels];
          float* out_pos = &output[(k * out_w * out_h + i * out_w + j) * channels];

          for (int c = 0; c < channels; ++c) {  // loop for channels
            // bilinear interpolation
            out_pos[c] = static_cast<float>(
                h2lambda * (w2lambda * in_pos[0 + c] + w1lambda * in_pos[wid + c]) +
                h1lambda * (w2lambda * in_pos[hid + c] +
                            w1lambda * in_pos[hid + wid + c]));
            // in_pos++;
            // out_pos++;
          }
        }
      }
    }
  }
}

template <>
bool BilinearInterpKernel<FPGA, float>::Init(BilinearInterpParam<FPGA> *param) {
	
  param->Out()->mutable_data<float>();

  return true;
}

template <>
void BilinearInterpKernel<FPGA, float>::Compute(
    const BilinearInterpParam<FPGA> &param) {

  
  BilinearInterpCompute<float>(param);
#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput(
      "bilinear_interp", param.Out()->zynqmpTensor());
#endif
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
