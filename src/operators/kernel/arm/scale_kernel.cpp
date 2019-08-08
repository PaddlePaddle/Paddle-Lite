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

#ifdef SCALE_OP

#include "operators/kernel/scale_kernel.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <>
bool ScaleKernel<CPU, float>::Init(ScaleParam<CPU> *param) {
  return true;
}

template <>
void ScaleKernel<CPU, float>::Compute(const ScaleParam<CPU> &param) {
  const auto input = param.InputX();
  auto output = param.Out();
  const float scale = param.Scale();
  const float bias = param.Bias();
  if (input->type() == type_id<int64_t>().hash_code()) {
    const int64_t *input_data = input->data<int64_t>();
    int64_t *output_data = output->mutable_data<int64_t>();

    int i = 0;
    for (; i < output->numel(); ++i, ++output_data, ++input_data) {
      *output_data = scale * (*input_data) + bias;
    }
  } else {
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    int i = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t vbias = vdupq_n_f32(bias);
    for (; i < output->numel() - 15; i += 16) {
      float32x4_t _in0 = vld1q_f32(input_data);
      float32x4_t _in1 = vld1q_f32(input_data + 4);
      float32x4_t _in2 = vld1q_f32(input_data + 8);
      float32x4_t _in3 = vld1q_f32(input_data + 12);
      _in0 = vmlaq_f32(vbias, vscale, _in0);
      _in1 = vmlaq_f32(vbias, vscale, _in1);
      _in2 = vmlaq_f32(vbias, vscale, _in2);
      _in3 = vmlaq_f32(vbias, vscale, _in3);
      vst1q_f32(output_data, _in0);
      vst1q_f32(output_data + 4, _in1);
      vst1q_f32(output_data + 8, _in2);
      vst1q_f32(output_data + 12, _in3);
      input_data += 16;
      output_data += 16;
    }
    for (; i < output->numel() - 3; i += 4) {
      float32x4_t _in0 = vld1q_f32(input_data);
      _in0 = vmlaq_f32(vbias, vscale, _in0);
      vst1q_f32(output_data, _in0);
      input_data += 4;
      output_data += 4;
    }
#endif
    for (; i < output->numel(); ++i, ++output_data, ++input_data) {
      *output_data = scale * (*input_data) + bias;
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
