/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/fpga/KD/pes/softmax_pe.hpp"

#include <vector>

namespace paddle {
namespace zynqmp {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#ifndef __aarch64__
static inline float32_t vmaxvq_f32(const float32x4_t &r) {
  float32x2_t v = vmax_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpmax_f32(v, v), 0);
}

static inline float32_t vaddvq_f32(const float32x4_t &r) {
  float32x2_t v = vadd_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpadd_f32(v, v), 0);
}
#endif  // __aarch64__
#endif  // __ARM_NEON__

static float find_max(const float *input, const int num_classes) {
  int remain = num_classes;
  float max = -std::numeric_limits<float>::max();
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int loop = num_classes >> 3;
  remain = num_classes & 0x7;
  float32x4_t __max = vdupq_n_f32(max);
  for (int i = 0; i < loop; ++i, input += 8) {
    float32x4_t x0 = vld1q_f32(input);
    float32x4_t x1 = vld1q_f32(input + 4);
    __max = vmaxq_f32(x0, __max);
    __max = vmaxq_f32(x1, __max);
  }
  max = vmaxvq_f32(__max);
#endif
  for (int i = 0; i < remain; ++i) {
    max = std::max(max, input[i]);
  }
  return max;
}

static void softmax(Tensor *X, Tensor *Y) {
  std::vector<int> dims = X->shape().dims();
  int batch_size = X->shape().num();
  int num_classes = dims[X->shape().dimSize() - 1];
  int channels = X->shape().numel() / batch_size / num_classes;

  float *x = X->data<float>();
  float *y = Y->mutableData<float>();

#pragma omp parallel for collapse(2)
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * num_classes;
      const float *input = x + offset;
      float *output = y + offset;
      // find max
      float max = find_max(input, num_classes);

      // exp(x - max)
      int remain = num_classes;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      int loop = num_classes >> 3;
      remain = num_classes & 0x7;
      float32x4_t __max = vdupq_n_f32(max);
      for (int i = 0; i < loop; ++i, input += 8, output += 8) {
        float32x4_t x0 = vld1q_f32(input);
        float32x4_t x1 = vld1q_f32(input + 4);
        x0 = vsubq_f32(x0, __max);
        x1 = vsubq_f32(x1, __max);
        x0 = lite::arm::math::exp_ps(x0);
        x1 = lite::arm::math::exp_ps(x1);
        vst1q_f32(output, x0);
        vst1q_f32(output + 4, x1);
      }
#endif  // __ARM_NEON__
      for (int i = 0; i < remain; ++i) {
        output[i] = expf(input[i] - max);
      }

      // sum(exp(x - max))
      float sum = 0.f;
      output = y + offset;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      float32x4_t __sum = vdupq_n_f32(0.f);
      for (int i = 0; i < loop; ++i, output += 8) {
        float32x4_t x0 = vld1q_f32(output);
        float32x4_t x1 = vld1q_f32(output + 4);
        __sum = vaddq_f32(x0, __sum);
        __sum = vaddq_f32(x1, __sum);
      }
      sum += vaddvq_f32(__sum);
#endif  // __ARM_NEON__
      for (int i = 0; i < remain; ++i) {
        sum += output[i];
      }

      // exp(x - max) / sum
      float inv_sum = 1.f / sum;
      output = y + offset;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      float32x4_t __inv_sum = vdupq_n_f32(inv_sum);
      for (int i = 0; i < loop; ++i, output += 8) {
        float32x4_t x0 = vld1q_f32(output);
        float32x4_t x1 = vld1q_f32(output + 4);
        x0 = vmulq_f32(x0, __inv_sum);
        x1 = vmulq_f32(x1, __inv_sum);
        vst1q_f32(output, x0);
        vst1q_f32(output + 4, x1);
      }
#endif
      for (int i = 0; i < remain; ++i) {
        output[i] *= inv_sum;
      }
    }
  }
}

bool SoftmaxPE::init() {
  Tensor *output = param_.output;
  output->setAligned(false);
  output->setDataLocation(CPU);
  return true;
}

bool SoftmaxPE::dispatch() {
  Tensor *input = param_.input;
  Tensor *output = param_.output;

  Tensor float_input;
  Tensor float_output;
  float_input.mutableData<float>(DataType::FP32, input->shape());
  input->syncToDevice();
  float_input.copyFrom(input);

  float *out_data =
      float_output.mutableData<float>(DataType::FP32, input->shape());

  softmax(&float_input, &float_output);
  float_output.flush();

  output->copyFrom(&float_output);
  output->flush();
  return true;
}

SoftmaxParam &SoftmaxPE::param() { return param_; }
}  // namespace zynqmp
}  // namespace paddle
