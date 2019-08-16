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

#ifdef NEAREST_INTERP_OP

#include "operators/kernel/nearest_interp_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool NearestInterpolationKernel<CPU, float>::Init(
    NearestInterpolationParam<CPU>* param) {
  return true;
}

template <>
void NearestInterpolationKernel<CPU, float>::Compute(
    const NearestInterpolationParam<CPU>& param) {
  auto out_dims = param.Out()->dims();
  auto* input = param.InputX()->data<float>();
  auto out_size_t = param.InputOutPutSize();

  int out_h = param.OutH();
  int out_w = param.OutW();
  if (out_size_t != nullptr) {
    auto out_size_data = out_size_t->data<int>();
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }
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
        int h = ratio_h * i + 0.5f;

        for (int j = 0; j < out_w; ++j) {
          int w = ratio_w * j + 0.5f;

          // calculate four position for bilinear interpolation
          const float* in_pos = &input[k * in_chw + h * in_w + w];
          float* out_pos = &output[k * out_chw + i * out_w + j];

          for (int c = 0; c < channels; ++c) {  // loop for channels
            // nearest interpolation
            out_pos[0] = in_pos[0];
            in_pos += in_hw;
            out_pos += out_hw;
          }
        }
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
