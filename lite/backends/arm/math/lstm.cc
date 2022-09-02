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

#include "lite/backends/arm/math/lstm.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void add_bias_rowwise(Tensor* input,
                      const Tensor* bias,
                      int start_w,
                      int end_w) {
  auto in_dim = input->dims();
  int width = input->numel() / in_dim[0];
  int w_adds = width < end_w ? width : end_w;
  float* i_data = input->mutable_data<float>();
  const float* b_data = bias->data<float>();
  for (int i = 0; i < in_dim[0]; ++i) {
    for (int w = start_w; w < w_adds; ++w) {
      i_data[w] += b_data[w];
    }
    i_data += width;
  }
}

void vector_dot(
    float* out, const float* in, const float* v1, int size, const float* v2) {
  int loop = size >> 2;
  int remain = size & 3;
  const float* in_ptr = in;
  float* out_ptr = out;
  const float* v1_ptr = v1;
  const float* v2_ptr = v2;
  for (int i = 0; i < loop; ++i) {
    float32x4_t in = vld1q_f32(in_ptr);
    float32x4_t data1 = vld1q_f32(v1_ptr);
    if (!v2) {
      // in_out * v1
      float32x4_t out = vmulq_f32(in, data1);
      vst1q_f32(out_ptr, out);
      in_ptr += 4;
      v1_ptr += 4;
      out_ptr += 4;
    } else {
      // in_out + v1 * v2
      float32x4_t data2 = vld1q_f32(v2_ptr);
      float32x4_t out = vmlaq_f32(in, data1, data2);
      vst1q_f32(out_ptr, out);
      in_ptr += 4;
      v1_ptr += 4;
      out_ptr += 4;
      v2_ptr += 4;
    }
  }
  for (int i = 0; i < remain; ++i) {
    if (!v2) {
      out_ptr[i] = in_ptr[i] * v1_ptr[i];
    } else {
      out_ptr[i] = in_ptr[i] + v1_ptr[i] * v2_ptr[i];
    }
  }
}

float* row_offset(Tensor& input, int start) {  // NOLINT
  auto in_dim = input.dims();
  int width = input.numel() / in_dim[0];
  int offset = start < in_dim[0] ? start * width : input.numel();
  return input.mutable_data<float>() + offset;
}

#ifdef ENABLE_ARM_FP16
void add_bias_rowwise_fp16(Tensor* input,
                           const Tensor* bias,
                           int start_w,
                           int end_w) {
  auto in_dim = input->dims();
  int width = input->numel() / in_dim[0];
  int w_adds = width < end_w ? width : end_w;
  float16_t* i_data = input->mutable_data<float16_t>();
  const float16_t* b_data = bias->data<float16_t>();
  for (int i = 0; i < in_dim[0]; ++i) {
    for (int w = start_w; w < w_adds; ++w) {
      i_data[w] += b_data[w];
    }
    i_data += width;
  }
}

void vector_dot_fp16(float16_t* out,
                     const float16_t* in,
                     const float16_t* v1,
                     int size,
                     const float16_t* v2) {
  int loop = size >> 3;
  int remain = size & 7;
  const float16_t* in_ptr = in;
  float16_t* out_ptr = out;
  const float16_t* v1_ptr = v1;
  const float16_t* v2_ptr = v2;
  for (int i = 0; i < loop; ++i) {
    float16x8_t in = vld1q_f16(in_ptr);
    float16x8_t data1 = vld1q_f16(v1_ptr);
    if (!v2) {
      // in_out * v1
      float16x8_t out = vmulq_f16(in, data1);
      vst1q_f16(out_ptr, out);
      in_ptr += 8;
      v1_ptr += 8;
      out_ptr += 8;
    } else {
      // in_out + v1 * v2
      float16x8_t data2 = vld1q_f16(v2_ptr);
      float16x8_t out = vfmaq_f16(in, data1, data2);
      vst1q_f16(out_ptr, out);
      in_ptr += 8;
      v1_ptr += 8;
      out_ptr += 8;
      v2_ptr += 8;
    }
  }
  for (int i = 0; i < remain; ++i) {
    if (!v2) {
      out_ptr[i] = in_ptr[i] * v1_ptr[i];
    } else {
      out_ptr[i] = in_ptr[i] + v1_ptr[i] * v2_ptr[i];
    }
  }
}

template <>
void activation<float16_t>(const float16_t* din,
                           float16_t* dout,
                           int size,
                           std::string act_str,
                           int threads) {
  if (act_str == "sigmoid") {
    fp16::act_sigmoid<float16_t>(din, dout, size, threads);
  } else if (act_str == "tanh") {
    fp16::act_tanh<float16_t>(din, dout, size, threads);
  } else if (act_str == "relu") {
    fp16::act_relu<float16_t>(din, dout, size, threads);
  } else {
    LOG(FATAL) << "unsupport fp16 activation " << act_str;
  }
}
template <>
void activation<float16_t>(const float16_t* din,
                           float16_t* dout,
                           int size,
                           lite_api::ActivationType act_type,
                           int threads) {
  switch (act_type) {
    case lite_api::ActivationType::kSigmoid:
      fp16::act_sigmoid<float16_t>(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kSigmoid_v2:
      fp16::act_sigmoid<float16_t>(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kTanh:
      fp16::act_tanh<float16_t>(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kTanh_v2:
      fp16::act_tanh<float16_t>(din, dout, size, threads);
      break;
    case lite_api::ActivationType::kRelu:
      fp16::act_relu<float16_t>(din, dout, size, threads);
      break;
    default:
      LOG(FATAL) << "unsupport fp16 activation type:"
                 << static_cast<int>(act_type);
      break;
  }
}
#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
