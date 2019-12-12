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

#include "lite/kernels/arm/instance_norm_compute.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void InstanceNormCompute::PrepareForRun() {}

void InstanceNormCompute::Run() {
  auto& param = this->Param<param_t>();
  const float* in = param.x->data<float>();
  const float* scale = param.scale->data<float>();
  const float* bias = param.bias->data<float>();
  float* out = param.out->mutable_data<float>();
  float* saved_mean = param.saved_mean->mutable_data<float>();
  float* saved_variance = param.saved_variance->mutable_data<float>();
  float epsilon = param.epsilon;

  int n = param.x->dims()[0];
  int c = param.x->dims()[1];
  int nc = n * c;
  int height = param.x->dims()[2];
  int width = param.x->dims()[3];
  int spatial_size = height * width;
// compute saved_mean and saved_variance
#pragma omp parallel for
  for (int i = 0; i < nc; ++i) {
    const float* in_p = in + i * spatial_size;
    float sum_spatial = 0.f;
    float summ_spatial = 0.f;
    for (int h = 0; h < height; ++h) {
      int w = width;
      float32x4_t sum0 = vdupq_n_f32(0.f);
      float32x4_t sum1 = vdupq_n_f32(0.f);
      float32x4_t sum2 = vdupq_n_f32(0.f);
      float32x4_t sum3 = vdupq_n_f32(0.f);
      float32x4_t summ0 = vdupq_n_f32(0.f);
      float32x4_t summ1 = vdupq_n_f32(0.f);
      float32x4_t summ2 = vdupq_n_f32(0.f);
      float32x4_t summ3 = vdupq_n_f32(0.f);
      float32x4_t in0, in1, in2, in3;
      for (; w > 15; w -= 16) {
        in0 = vld1q_f32(in_p);
        in1 = vld1q_f32(in_p + 4);
        in2 = vld1q_f32(in_p + 8);
        in3 = vld1q_f32(in_p + 12);
        sum0 = vaddq_f32(sum0, in0);
        sum1 = vaddq_f32(sum1, in1);
        summ0 = vmlaq_f32(summ0, in0, in0);
        summ1 = vmlaq_f32(summ1, in1, in1);
        sum2 = vaddq_f32(sum2, in2);
        sum3 = vaddq_f32(sum3, in3);
        summ2 = vmlaq_f32(summ2, in2, in2);
        summ3 = vmlaq_f32(summ3, in3, in3);
        in_p += 16;
      }
      for (; w > 7; w -= 8) {
        in0 = vld1q_f32(in_p);
        in1 = vld1q_f32(in_p + 4);
        sum0 = vaddq_f32(sum0, in0);
        sum1 = vaddq_f32(sum1, in1);
        summ0 = vmlaq_f32(summ0, in0, in0);
        summ1 = vmlaq_f32(summ1, in1, in1);
        in_p += 8;
      }
      for (; w > 3; w -= 4) {
        in0 = vld1q_f32(in_p);
        sum0 = vaddq_f32(sum0, in0);
        summ0 = vmlaq_f32(summ0, in0, in0);
        in_p += 4;
      }
      float sum = 0.f;
      float summ = 0.f;
      for (; w > 0; w--) {
        sum += *in_p;
        summ += (*in_p) * (*in_p);
        in_p++;
      }
      sum0 = vaddq_f32(sum0, sum1);
      sum2 = vaddq_f32(sum2, sum3);
      summ0 = vaddq_f32(summ0, summ1);
      summ2 = vaddq_f32(summ2, summ3);
      sum0 = vaddq_f32(sum0, sum2);
      summ0 = vaddq_f32(summ0, summ2);
      float32x2_t sum_low = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
      float32x2_t sum_high =
          vpadd_f32(vget_low_f32(summ0), vget_high_f32(summ0));
      float32x2_t sum_mix = vpadd_f32(sum_low, sum_high);
      sum += vget_lane_f32(sum_mix, 0);
      summ += vget_lane_f32(sum_mix, 1);
      sum_spatial += sum;
      summ_spatial += summ;
    }
    float mean = sum_spatial / spatial_size;
    // float variance = summ / spatial_size - mean * mean;
    // the flolowing code has higher precision than above comment code
    float variance = (summ_spatial - mean * mean * spatial_size) / spatial_size;
    float std = 1.f / sqrtf(variance + epsilon);

    saved_mean[i] = mean;
    saved_variance[i] = std;
  }
// compute instance_norm result: out = scale * (in - mean) / std + bias
#pragma omp parallel for
  for (int i = 0; i < nc; ++i) {
    const float* in_p = in + i * spatial_size;
    float* out_p = out + i * spatial_size;
    int j = spatial_size;
    const float sstd_val = scale[i % c] * saved_variance[i];
    const float bias_val = bias[i % c];
    const float mean_val = saved_mean[i];
    const float32x4_t vsstd = vdupq_n_f32(sstd_val);
    const float32x4_t vbias = vdupq_n_f32(bias_val);
    const float32x4_t vmean = vdupq_n_f32(mean_val);
    float32x4_t in0, in1, submean0, submean1, out0, out1;
    for (; j > 7; j -= 8) {
      in0 = vld1q_f32(in_p);
      in1 = vld1q_f32(in_p + 4);
      submean0 = vsubq_f32(in0, vmean);
      submean1 = vsubq_f32(in1, vmean);
      out0 = vmlaq_f32(vbias, submean0, vsstd);
      out1 = vmlaq_f32(vbias, submean1, vsstd);
      vst1q_f32(out_p, out0);
      vst1q_f32(out_p + 4, out1);
      in_p += 8;
      out_p += 8;
    }
    for (; j > 3; j -= 4) {
      in0 = vld1q_f32(in_p);
      submean0 = vsubq_f32(in0, vmean);
      out0 = vmlaq_f32(vbias, submean0, vsstd);
      vst1q_f32(out_p, out0);
      in_p += 4;
      out_p += 4;
    }
    for (; j > 0; j--) {
      *out_p = (*in_p - mean_val) * sstd_val + bias_val;
      in_p++;
      out_p++;
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(instance_norm,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::InstanceNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
