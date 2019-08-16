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

#pragma once

#include "framework/tensor.h"
#include "operators/math/activation.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

template <ActivationType Act>
void AddChannelWise(const framework::Tensor *input,
                    const framework::Tensor *bias, framework::Tensor *output) {
  const float *input_ptr = input->data<float>();
  const float *bias_ptr = bias->data<float>();
  float *output_ptr = output->mutable_data<float>();
  // maybe check shape
  int batch_size = input->dims()[0];
  int channels = input->dims()[1];
  int spatial_size = input->dims()[2] * input->dims()[3];

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * spatial_size;
      const float *x = input_ptr + offset;
      float *y = output_ptr + offset;
      float beta = bias_ptr[channel];
      int j = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t __bias = vdupq_n_f32(beta);
      for (; j < spatial_size - 15; j += 16, x += 16, y += 16) {
        float32x4_t in0 = vld1q_f32(x);
        float32x4_t in1 = vld1q_f32(x + 4);
        float32x4_t in2 = vld1q_f32(x + 8);
        float32x4_t in3 = vld1q_f32(x + 12);
        in0 = vaddq_f32(__bias, in0);
        in1 = vaddq_f32(__bias, in1);
        in2 = vaddq_f32(__bias, in2);
        in3 = vaddq_f32(__bias, in3);
        in0 = math::vActiveq_f32<Act>(in0);
        in1 = math::vActiveq_f32<Act>(in1);
        in2 = math::vActiveq_f32<Act>(in2);
        in3 = math::vActiveq_f32<Act>(in3);
        vst1q_f32(y, in0);
        vst1q_f32(y + 4, in1);
        vst1q_f32(y + 8, in2);
        vst1q_f32(y + 12, in3);
      }
      for (; j < spatial_size - 3; j += 4, x += 4, y += 4) {
        float32x4_t in0 = vld1q_f32(x);
        in0 = vaddq_f32(__bias, in0);
        in0 = math::vActiveq_f32<Act>(in0);
        vst1q_f32(y, in0);
      }
#endif
      for (; j < spatial_size; ++j, ++x, ++y) {
        *y = math::Active<Act>((*x) + beta);
      }
    }
  }
}

template <ActivationType Act>
void ScaleAddChannelWise(const framework::Tensor *input,
                         const framework::Tensor *scale,
                         const framework::Tensor *bias,
                         framework::Tensor *output) {
  const float *input_ptr = input->data<float>();
  const float *scale_ptr = scale->data<float>();
  const float *bias_ptr = bias->data<float>();
  float *output_ptr = output->mutable_data<float>();
  // maybe check shape
  int batch_size = input->dims()[0];
  int channels = input->dims()[1];
  int spatial_size = input->dims()[2] * input->dims()[3];

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * spatial_size;
      const float *x = input_ptr + offset;
      float *y = output_ptr + offset;
      float alpha = scale_ptr[channel];
      float beta = bias_ptr[channel];
      int j = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t __scale = vdupq_n_f32(alpha);
      float32x4_t __bias = vdupq_n_f32(beta);
      for (; j < spatial_size - 15; j += 16, x += 16, y += 16) {
        float32x4_t in0 = vld1q_f32(x);
        float32x4_t in1 = vld1q_f32(x + 4);
        float32x4_t in2 = vld1q_f32(x + 8);
        float32x4_t in3 = vld1q_f32(x + 12);
        in0 = vmlaq_f32(__bias, __scale, in0);
        in1 = vmlaq_f32(__bias, __scale, in1);
        in2 = vmlaq_f32(__bias, __scale, in2);
        in3 = vmlaq_f32(__bias, __scale, in3);
        in0 = math::vActiveq_f32<Act>(in0);
        in1 = math::vActiveq_f32<Act>(in1);
        in2 = math::vActiveq_f32<Act>(in2);
        in3 = math::vActiveq_f32<Act>(in3);
        vst1q_f32(y, in0);
        vst1q_f32(y + 4, in1);
        vst1q_f32(y + 8, in2);
        vst1q_f32(y + 12, in3);
      }
      for (; j < spatial_size - 3; j += 4, x += 4, y += 4) {
        float32x4_t in0 = vld1q_f32(x);
        in0 = vmlaq_f32(__bias, __scale, in0);
        in0 = math::vActiveq_f32<Act>(in0);
        vst1q_f32(y, in0);
      }
#endif
      for (; j < spatial_size; ++j, ++x, ++y) {
        *y = math::Active<Act>(alpha * (*x) + beta);
      }
    }
  }
}

template <ActivationType Act>
void ScaleAddChannelWise(const framework::Tensor *input,
                         const framework::Tensor *scale,
                         const framework::Tensor *bias,
                         const framework::Tensor *tensorwise_bias,
                         framework::Tensor *output) {
  const float *input_ptr = input->data<float>();
  const float *scale_ptr = scale->data<float>();
  const float *bias_ptr = bias->data<float>();
  const float *tensorwise_bias_ptr = tensorwise_bias->data<float>();
  float *output_ptr = output->mutable_data<float>();
  // maybe check shape
  int batch_size = input->dims()[0];
  int channels = input->dims()[1];
  int spatial_size = input->dims()[2] * input->dims()[3];

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * spatial_size;
      const float *x = input_ptr + offset;
      const float *b = tensorwise_bias_ptr + offset;
      float *y = output_ptr + offset;
      float alpha = scale_ptr[channel];
      float beta = bias_ptr[channel];
      int j = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t __scale = vdupq_n_f32(alpha);
      float32x4_t __bias = vdupq_n_f32(beta);
      for (; j < spatial_size - 15; j += 16, x += 16, b += 16, y += 16) {
        float32x4_t in0 = vld1q_f32(x);
        float32x4_t in1 = vld1q_f32(x + 4);
        float32x4_t in2 = vld1q_f32(x + 8);
        float32x4_t in3 = vld1q_f32(x + 12);
        float32x4_t b0 = vld1q_f32(b);
        float32x4_t b1 = vld1q_f32(b + 4);
        float32x4_t b2 = vld1q_f32(b + 8);
        float32x4_t b3 = vld1q_f32(b + 12);
        in0 = vmlaq_f32(__bias, __scale, in0);
        in1 = vmlaq_f32(__bias, __scale, in1);
        in2 = vmlaq_f32(__bias, __scale, in2);
        in3 = vmlaq_f32(__bias, __scale, in3);
        in0 = vaddq_f32(in0, b0);
        in1 = vaddq_f32(in1, b1);
        in2 = vaddq_f32(in2, b2);
        in3 = vaddq_f32(in3, b3);
        in0 = math::vActiveq_f32<Act>(in0);
        in1 = math::vActiveq_f32<Act>(in1);
        in2 = math::vActiveq_f32<Act>(in2);
        in3 = math::vActiveq_f32<Act>(in3);
        vst1q_f32(y, in0);
        vst1q_f32(y + 4, in1);
        vst1q_f32(y + 8, in2);
        vst1q_f32(y + 12, in3);
      }
      for (; j < spatial_size - 3; j += 4, x += 4, b += 4, y += 4) {
        float32x4_t in0 = vld1q_f32(x);
        float32x4_t b0 = vld1q_f32(b);
        in0 = vmlaq_f32(__bias, __scale, in0);
        in0 = vaddq_f32(in0, b0);
        in0 = math::vActiveq_f32<Act>(in0);
        vst1q_f32(y, in0);
      }
#endif
      for (; j < spatial_size; ++j, ++x, ++b, ++y) {
        *y = math::Active<Act>(alpha * (*x) + beta + (*b));
      }
    }
  }
}

template <ActivationType Act>
void AddElememtWise(const framework::Tensor *input,
                    const framework::Tensor *bias, const int axis,
                    framework::Tensor *output) {
  const auto &x_dims = input->dims();
  const auto &y_dims = bias->dims();
  const float *input_data = input->data<float>();
  const float *bias_data = bias->data<float>();
  float *output_data = output->mutable_data<float>();

  if (x_dims == y_dims) {
    size_t channels = 1;
    size_t elementwise_num = 1;
    for (int i = 0; i < y_dims.size(); ++i) {
      channels *= y_dims[i];
    }
#pragma omp parallel for
    for (int j = 0; j < channels; ++j) {
      size_t offset = (0 * channels + j) * elementwise_num;
      const float *input = input_data + offset;
      const float bias = bias_data[j];
      float *output = output_data + offset;
#if 0
      int loop = elementwise_num >> 0x4;
      int remain = elementwise_num & 0xF;
      float32x4_t rb = vdupq_n_f32(bias);
      for (int k = 0; k < loop; ++k) {
        float32x4_t r0 = vld1q_f32(input);
        float32x4_t r1 = vld1q_f32(input + 4);
        float32x4_t r2 = vld1q_f32(input + 8);
        float32x4_t r3 = vld1q_f32(input + 12);
        r0 = vaddq_f32(r0, rb);
        r1 = vaddq_f32(r1, rb);
        r2 = vaddq_f32(r2, rb);
        r3 = vaddq_f32(r3, rb);
        r0 = math::vActiveq_f32<Act>(r0);
        r1 = math::vActiveq_f32<Act>(r1);
        r2 = math::vActiveq_f32<Act>(r2);
        r3 = math::vActiveq_f32<Act>(r3);
        vst1q_f32(output, r0);
        vst1q_f32(output + 4, r1);
        vst1q_f32(output + 8, r2);
        vst1q_f32(output + 12, r3);
        input += 16;
        output += 16;
      }
      if (remain >= 8) {
        float32x4_t r0 = vld1q_f32(input);
        float32x4_t r1 = vld1q_f32(input + 4);
        r0 = vaddq_f32(r0, rb);
        r1 = vaddq_f32(r1, rb);
        r0 = math::vActiveq_f32<Act>(r0);
        r1 = math::vActiveq_f32<Act>(r1);
        vst1q_f32(output, r0);
        vst1q_f32(output + 4, r1);
        input += 8;
        output += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t r0 = vld1q_f32(input);
        r0 = vaddq_f32(r0, rb);
        r0 = math::vActiveq_f32<Act>(r0);
        vst1q_f32(output, r0);
        input += 4;
        output += 4;
        remain -= 4;
      }
      if (remain > 0) {
        float32x4_t r0 = vld1q_f32(input);
        r0 = vaddq_f32(r0, rb);
        r0 = math::vActiveq_f32<Act>(r0);
        switch (remain) {
          case 1:
            vst1q_lane_f32(output, r0, 0);
            break;
          case 2:
            vst1_f32(output, vget_low_f32(r0));
            break;
          case 3:
            vst1_f32(output, vget_low_f32(r0));
            vst1q_lane_f32(output, r0, 2);
            break;
        }
      }
#else
      for (int k = 0; k < elementwise_num; ++k) {
        output[k] = math::Active<Act>(input[k] + bias);
      }
#endif  // __ARM_NEON__
    }

  } else {
    // axis = -1 represent the last dimensions.
    int dim = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
    size_t batch = 1;
    size_t channels = 1;
    size_t elementwise_num = 1;
    for (int i = 0; i < dim; ++i) {
      batch *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      channels *= y_dims[i];
    }
    for (int i = y_dims.size() + dim; i < x_dims.size(); ++i) {
      elementwise_num *= x_dims[i];
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        size_t offset = (i * channels + j) * elementwise_num;
        const float *input = input_data + offset;
        const float bias = bias_data[j];
        float *output = output_data + offset;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        int loop = elementwise_num >> 0x4;
        int remain = elementwise_num & 0xF;
        float32x4_t rb = vdupq_n_f32(bias);
        for (int k = 0; k < loop; ++k) {
          float32x4_t r0 = vld1q_f32(input);
          float32x4_t r1 = vld1q_f32(input + 4);
          float32x4_t r2 = vld1q_f32(input + 8);
          float32x4_t r3 = vld1q_f32(input + 12);
          r0 = vaddq_f32(r0, rb);
          r1 = vaddq_f32(r1, rb);
          r2 = vaddq_f32(r2, rb);
          r3 = vaddq_f32(r3, rb);
          r0 = math::vActiveq_f32<Act>(r0);
          r1 = math::vActiveq_f32<Act>(r1);
          r2 = math::vActiveq_f32<Act>(r2);
          r3 = math::vActiveq_f32<Act>(r3);
          vst1q_f32(output, r0);
          vst1q_f32(output + 4, r1);
          vst1q_f32(output + 8, r2);
          vst1q_f32(output + 12, r3);
          input += 16;
          output += 16;
        }
        if (remain >= 8) {
          float32x4_t r0 = vld1q_f32(input);
          float32x4_t r1 = vld1q_f32(input + 4);
          r0 = vaddq_f32(r0, rb);
          r1 = vaddq_f32(r1, rb);
          r0 = math::vActiveq_f32<Act>(r0);
          r1 = math::vActiveq_f32<Act>(r1);
          vst1q_f32(output, r0);
          vst1q_f32(output + 4, r1);
          input += 8;
          output += 8;
          remain -= 8;
        }
        if (remain >= 4) {
          float32x4_t r0 = vld1q_f32(input);
          r0 = vaddq_f32(r0, rb);
          r0 = math::vActiveq_f32<Act>(r0);
          vst1q_f32(output, r0);
          input += 4;
          output += 4;
          remain -= 4;
        }
        if (remain > 0) {
          float32x4_t r0 = vld1q_f32(input);
          r0 = vaddq_f32(r0, rb);
          r0 = math::vActiveq_f32<Act>(r0);
          switch (remain) {
            case 1:
              vst1q_lane_f32(output, r0, 0);
              break;
            case 2:
              vst1_f32(output, vget_low_f32(r0));
              break;
            case 3:
              vst1_f32(output, vget_low_f32(r0));
              vst1q_lane_f32(output, r0, 2);
              break;
          }
        }
#else
        for (int k = 0; k < elementwise_num; ++k) {
          output[k] = math::Active<Act>(input[k] + bias);
        }
#endif  // __ARM_NEON__
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
