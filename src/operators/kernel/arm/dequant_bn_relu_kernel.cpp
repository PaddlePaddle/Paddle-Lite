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

#include "operators/kernel/dequant_bn_relu_kernel.h"
#include <cmath>
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

#if defined(FUSION_DEQUANT_BN_RELU_OP) || defined(FUSION_DEQUANT_ADD_BN_RELU_OP)
void DequantBNReluCompute(const FusionDequantBNParam<CPU> *param) {
  const int32_t *input = param->input_->data<int32_t>();
  const float *bn_scale = param->bn_scale_->data<float>();
  const float *bn_bias = param->bn_bias_->data<float>();
  // dequantize params
  const float activation_scale = param->activation_scale_->data<float>()[0];
  const float weight_scale = param->weight_scale_;
  const float dequant_scale = activation_scale / weight_scale;

  float *output = param->output_->mutable_data<float>();
  int batch_size = param->input_->dims()[0];
  int channels = param->input_->dims()[1];
  size_t spatial_size = param->input_->dims()[2] * param->input_->dims()[3];

  #pragma omp parallel for collapse(2)
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int c = 0; c < channels; ++c) {
      float scale = bn_scale[c] * dequant_scale;
      float bias = bn_bias[c];
      size_t offset = (batch * channels + c) * spatial_size;
      const int32_t *x = input + offset;
      float *y = output + offset;
      size_t remain = spatial_size;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop = spatial_size >> 4;
      remain = spatial_size & 0xF;
      float32x4_t __scale = vdupq_n_f32(scale);
      float32x4_t __bias = vdupq_n_f32(bias);
      float32x4_t __zero = vdupq_n_f32(0.f);

      for (int k = 0; k < loop; ++k, x += 16, y += 16) {
        int32x4_t r0 = vld1q_s32(x);
        int32x4_t r1 = vld1q_s32(x + 4);
        int32x4_t r2 = vld1q_s32(x + 8);
        int32x4_t r3 = vld1q_s32(x + 12);
        float32x4_t f0 = vcvtq_f32_s32(r0);
        float32x4_t f1 = vcvtq_f32_s32(r1);
        float32x4_t f2 = vcvtq_f32_s32(r2);
        float32x4_t f3 = vcvtq_f32_s32(r3);
        f0 = vmlaq_f32(__bias, __scale, f0);
        f1 = vmlaq_f32(__bias, __scale, f1);
        f2 = vmlaq_f32(__bias, __scale, f2);
        f3 = vmlaq_f32(__bias, __scale, f3);
        f0 = vmaxq_f32(__zero, f0);
        f1 = vmaxq_f32(__zero, f1);
        f2 = vmaxq_f32(__zero, f2);
        f3 = vmaxq_f32(__zero, f3);
        vst1q_f32(y, f0);
        vst1q_f32(y + 4, f1);
        vst1q_f32(y + 8, f2);
        vst1q_f32(y + 12, f3);
      }
#endif  // __ARM_NEON__
      for (int k = 0; k < remain; ++k) {
        y[k] = std::max(scale * x[k] + bias, 0.f);
      }
    }
  }
}
#endif

#ifdef FUSION_DEQUANT_BN_RELU_OP
template <>
bool FusionDequantBNReluKernel<CPU, float>::Init(
    FusionDequantBNReluParam<CPU> *param) {
  // batch norm params
  const Tensor *bn_mean = param->bn_mean_;
  const Tensor *bn_variance = param->bn_variance_;
  Tensor *bn_scale = param->bn_scale_;
  Tensor *bn_bias = param->bn_bias_;
  const float epsilon = param->epsilon_;

  const float *mean_ptr = bn_mean->data<float>();
  const float *var_ptr = bn_variance->data<float>();
  float *bn_scale_ptr = bn_scale->mutable_data<float>();
  float *bn_bias_ptr = bn_bias->mutable_data<float>();
  for (int c = 0; c < bn_scale->numel(); ++c) {
    float inv_scale = bn_scale_ptr[c] / (std::sqrt(var_ptr[c] + epsilon));
    bn_scale_ptr[c] = inv_scale;
    bn_bias_ptr[c] = bn_bias_ptr[c] - inv_scale * mean_ptr[c];
  }
  return true;
}

template <>
void FusionDequantBNReluKernel<CPU, float>::Compute(
    const FusionDequantBNReluParam<CPU> &param) {
  DequantBNReluCompute(&param);
}
#endif  // FUSION_DEQUANT_BN_RELU_OP

#ifdef FUSION_DEQUANT_ADD_BN_RELU_OP
template <>
bool FusionDequantAddBNReluKernel<CPU, float>::Init(
    FusionDequantAddBNReluParam<CPU> *param) {
  // elementwise add params
  const Tensor *bias = param->bias_;
  // batch norm params
  const Tensor *bn_mean = param->bn_mean_;
  const Tensor *bn_variance = param->bn_variance_;
  Tensor *bn_scale = param->bn_scale_;
  Tensor *bn_bias = param->bn_bias_;
  const float epsilon = param->epsilon_;

  const float *bias_ptr = bias->data<float>();
  const float *mean_ptr = bn_mean->data<float>();
  const float *var_ptr = bn_variance->data<float>();
  float *bn_scale_ptr = bn_scale->mutable_data<float>();
  float *bn_bias_ptr = bn_bias->mutable_data<float>();
  for (int c = 0; c < bn_scale->numel(); ++c) {
    float inv_scale = bn_scale_ptr[c] / (std::sqrt(var_ptr[c] + epsilon));
    bn_scale_ptr[c] = inv_scale;
    bn_bias_ptr[c] = inv_scale * (bias_ptr[c] - mean_ptr[c]) + bn_bias_ptr[c];
  }
  return true;
}

template <>
void FusionDequantAddBNReluKernel<CPU, float>::Compute(
    const FusionDequantAddBNReluParam<CPU> &param) {
  DequantBNReluCompute(&param);
}
#endif  // FUSION_DEQUANT_ADD_BN_RELU_OP

}  // namespace operators
}  // namespace paddle_mobile
