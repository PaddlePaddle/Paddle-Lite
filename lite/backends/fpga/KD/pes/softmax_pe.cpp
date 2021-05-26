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

void fpga_softmax(int axis,
                  Tensor *input,
                  Tensor *output,
                  PoolingPE *poolingPE_) {
  Tensor norm_exp_output;
  norm_exp_output.mutableData<float16>(FP16, output->shape());
  norm_exp_output.shape().setLayoutType(output->shape().getLayoutType());

  PoolingArgs norm_exp_args = {0};
  norm_exp_args.mode = 0x4;
  norm_exp_args.kernel_reciprocal = 1.0f;
  norm_exp_args.image.address = input->data<void>();
  norm_exp_args.image.channels = input->shape().channel();
  norm_exp_args.image.height = input->shape().height();
  norm_exp_args.image.width = input->shape().width();
  norm_exp_args.image.pad_height = 0;
  norm_exp_args.image.pad_width = 0;
  norm_exp_args.image.scale_address = input->max();
  norm_exp_args.output.address = norm_exp_output.mutableData<float16>();
  norm_exp_args.output.scale_address = norm_exp_output.max();
  norm_exp_args.kernel.height = 1;
  norm_exp_args.kernel.width = 1;
  norm_exp_args.kernel.stride_h = 1;
  norm_exp_args.kernel.stride_w = 1;
  norm_exp_args.out_height = norm_exp_output.shape().height();
  norm_exp_args.out_width = norm_exp_output.shape().width();

  input->flush();

  compute_fpga_pool(norm_exp_args);

  norm_exp_output.invalidate();

  PoolingArgs prob_args = {0};
  prob_args.mode = 0x8;
  prob_args.kernel_reciprocal = 1.0f;
  prob_args.image.address = norm_exp_output.data<float16>();
  prob_args.image.channels = norm_exp_output.shape().channel();
  prob_args.image.height = norm_exp_output.shape().height();
  prob_args.image.width = norm_exp_output.shape().width();
  prob_args.image.pad_height = 0;
  prob_args.image.pad_width = 0;
  prob_args.image.scale_address = norm_exp_output.max();
  prob_args.output.address = output->mutableData<float16>();
  prob_args.output.scale_address = output->max();
  prob_args.kernel.height = 1;
  prob_args.kernel.width = 1;
  prob_args.kernel.stride_h = 1;
  prob_args.kernel.stride_w = 1;
  prob_args.out_height = output->shape().height();
  prob_args.out_width = output->shape().width();

  compute_fpga_pool(prob_args);
  struct FpgaRegWriteArgs args;
  args.value = 0;
  args.address = 0x890;
  output->invalidate();
}

bool SoftmaxPE::init() {
  Tensor *output = param_.output;
  output->setAligned(false);
  output->setDataLocation(CPU);
  return true;
}

void SoftmaxPE::apply() { use_cpu_ = param_.input->shape().dimSize() <= 2; }

bool SoftmaxPE::dispatch() {
  Tensor *input = param_.input;
  Tensor *output = param_.output;
  int axis = param_.axis;

  if (use_cpu_) {
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
  } else {
    fpga_softmax(axis, input, output, &poolingPE_);
  }

  return true;
}

SoftmaxParam &SoftmaxPE::param() { return param_; }
}  // namespace zynqmp
}  // namespace paddle
