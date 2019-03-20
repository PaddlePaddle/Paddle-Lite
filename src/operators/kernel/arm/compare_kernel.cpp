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

#include "operators/kernel/compare_kernel.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

typedef enum {
  LESS_THAN = 0,
  LESS_EQUAL = 1,
  GREATER_THAN = 2,
  GREATER_EQUAL = 3,
  EQUAL = 4,
  NOT_EQUAL = 5,
} CompareType;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <CompareType Comp = LESS_THAN>
inline uint32x4_t vcmpq_f32(const float32x4_t x, const float32x4_t y) {
  return vcleq_f32(x, y);
}
#endif

template <CompareType Comp = LESS_THAN>
inline uint8_t Compare(const float x, const float y) {
  return static_cast<uint8_t>(x < y);
}

template <CompareType Comp = LESS_THAN>
inline uint8_t Compare(const int64_t x, const int64_t y) {
  return static_cast<uint8_t>(x < y);
}

template <typename Dtype, CompareType Comp>
struct CompareCompute {
  void operator()(const Tensor *X, const Tensor *Y, const int Axis,
                  Tensor *Out) {}
};

template <CompareType Comp>
struct CompareCompute<float, Comp> {
  void operator()(const Tensor *X, const Tensor *Y, const int Axis,
                  Tensor *Out) {
    const float *x = X->data<float>();
    const float *y = Y->data<float>();
    uint8_t *output = reinterpret_cast<uint8_t *>(Out->mutable_data<bool>());
    const auto &x_dims = X->dims();
    const auto &y_dims = Y->dims();
    /// axis = -1 represent the last dimensions.
    int axis = (Axis == -1 ? x_dims.size() - y_dims.size() : Axis);
    int batch = 1;
    int channels = 1;
    int elementwise_num = 1;
    for (int i = 0; i < axis; ++i) {
      batch *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      channels *= y_dims[i];
    }
    for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
      elementwise_num *= x_dims[i];
    }
    // if elementwise_num == 1, compare rowwise
    if (elementwise_num == 1) {
      int remain_start = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      remain_start = channels & 0xfffffff8;
      uint8x8_t __mask = vdup_n_u8(0x1);
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels - 7; j += 8) {
          int x_offset = i * channels + j;
          float32x4_t __x0 = vld1q_f32(x + x_offset);
          float32x4_t __x1 = vld1q_f32(x + x_offset + 4);
          float32x4_t __y0 = vld1q_f32(y + j);
          float32x4_t __y1 = vld1q_f32(y + j + 4);
          uint32x4_t __cmp0 = vcmpq_f32<Comp>(__x0, __y0);
          uint32x4_t __cmp1 = vcmpq_f32<Comp>(__x1, __y1);
          uint16x4_t __ncmp0 = vmovn_u32(__cmp0);
          uint16x4_t __ncmp1 = vmovn_u32(__cmp1);
          uint16x8_t __ncmp = vcombine_u16(__ncmp0, __ncmp1);
          uint8x8_t __nncmp = vmovn_u16(__ncmp);
          __nncmp = vand_u8(__nncmp, __mask);
          vst1_u8(output + x_offset, __nncmp);
        }
      }
#endif  // __ARM_NEON__
      for (int i = 0; i < batch; ++i) {
        for (int j = remain_start; j < channels; ++j) {
          int x_offset = i * channels + j;
          output[x_offset] = Compare<Comp>(x[x_offset], y[j]);
        }
      }
    } else {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          int x_offset = (i * channels + j) * elementwise_num;
          int y_offset = j * elementwise_num;
          int remain_start = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
          remain_start = elementwise_num & 0xfffffff8;
          uint8x8_t __mask = vdup_n_u8(0x1);
          for (int k = 0; k < elementwise_num - 7; k += 8) {
            float32x4_t __x0 = vld1q_f32(x + x_offset);
            float32x4_t __x1 = vld1q_f32(x + x_offset + 4);
            float32x4_t __y0 = vld1q_f32(y + y_offset);
            uint32x4_t __cmp0 = vcmpq_f32<Comp>(__x0, __y0);
            uint32x4_t __cmp1 = vcmpq_f32<Comp>(__x1, __y0);
            uint16x4_t __ncmp0 = vmovn_u32(__cmp0);
            uint16x4_t __ncmp1 = vmovn_u32(__cmp1);
            uint16x8_t __ncmp = vcombine_u16(__ncmp0, __ncmp1);
            uint8x8_t __nncmp = vmovn_u16(__ncmp);
            __nncmp = vand_u8(__nncmp, __mask);
            vst1_u8(output + x_offset, __nncmp);
            x_offset += 8;
            y_offset += 8;
          }
#endif  // __ARM_NEON__
          for (int k = remain_start; k < elementwise_num; ++k) {
            output[x_offset + k] = Compare<Comp>(x[x_offset + k], y[y_offset]);
          }
        }
      }
    }
  }
};

template <CompareType Comp>
struct CompareCompute<int64_t, Comp> {
  void operator()(const Tensor *X, const Tensor *Y, const int Axis,
                  Tensor *Out) {
    const int64_t *x = X->data<int64_t>();
    const int64_t *y = Y->data<int64_t>();
    uint8_t *output = reinterpret_cast<uint8_t *>(Out->mutable_data<bool>());
    const auto &x_dims = X->dims();
    const auto &y_dims = Y->dims();
    /// axis = -1 represent the last dimensions.
    int axis = (Axis == -1 ? x_dims.size() - y_dims.size() : Axis);
    int batch = 1;
    int channels = 1;
    int elementwise_num = 1;
    for (int i = 0; i < axis; ++i) {
      batch *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      channels *= y_dims[i];
    }
    for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
      elementwise_num *= x_dims[i];
    }
    // if elementwise_num == 1, compare rowwise
    if (elementwise_num == 1) {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          int x_offset = i * channels + j;
          output[x_offset] = Compare<Comp>(x[x_offset], y[j]);
        }
      }
    } else {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          int x_offset = (i * channels + j) * elementwise_num;
          int y_offset = j * elementwise_num;
          for (int k = 0; k < elementwise_num; ++k) {
            output[x_offset + k] = Compare<Comp>(x[x_offset + k], y[y_offset]);
          }
        }
      }
    }
  }
};

#ifdef LESS_THAN_OP
template <>
bool LessThanKernel<CPU, float>::Init(CompareParam<CPU> *param) {
  return true;
}

template <>
void LessThanKernel<CPU, float>::Compute(const CompareParam<CPU> &param) {
  if (param.input_x_->type() == typeid(int64_t)) {
    CompareCompute<int64_t, LESS_THAN>()(param.input_x_, param.input_y_,
                                         param.axis_, param.output_);
  } else if (param.input_x_->type() == typeid(float)) {
    CompareCompute<float, LESS_THAN>()(param.input_x_, param.input_y_,
                                       param.axis_, param.output_);
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(
        "LessThan only support int64_t and float data type.");
  }
}
#endif  // LESS_THAN_OP

}  // namespace operators
}  // namespace paddle_mobile
