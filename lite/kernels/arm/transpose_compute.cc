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

#include "lite/kernels/arm/transpose_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/parallel_defines.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename Dtype>
void transpose_mat(const Dtype* din,
                   Dtype* dout,
                   const int num,
                   const int width,
                   const int height);
#define INIT_PTR_4(dtype, ptr_out, size_h)            \
  dtype* data_out_ptr = ptr_out + w * size_h + tmp_h; \
  const dtype* din0 = ptr_din_row;                    \
  const dtype* din1 = din0 + width;                   \
  const dtype* din2 = din1 + width;                   \
  const dtype* din3 = din2 + width;                   \
  dtype* dout0 = data_out_ptr;                        \
  dtype* dout1 = dout0 + height;                      \
  dtype* dout2 = dout1 + height;                      \
  dtype* dout3 = dout2 + height;

#define INIT_PTR_A4(dtype)          \
  const dtype* din4 = din3 + width; \
  const dtype* din5 = din4 + width; \
  const dtype* din6 = din5 + width; \
  const dtype* din7 = din6 + width;

#define INIT_PTR_B4(dtype)       \
  dtype* dout4 = dout3 + height; \
  dtype* dout5 = dout4 + height; \
  dtype* dout6 = dout5 + height; \
  dtype* dout7 = dout6 + height;

void transpose_mat(const float* din,
                   float* dout,
                   const int num,
                   const int width,
                   const int height) {
  int nw = width >> 2;
  int nh = height >> 2;
  int size_in = width * height;
  int size_w = width << 2;
  int size_h = height << 2;

  for (int i = 0; i < num; ++i) {
    float* ptr_out = dout + i * size_in;
    const float* ptr_in = din + i * size_in;
    LITE_PARALLEL_BEGIN(h, tid, nh) {
      const float* ptr_din_row = ptr_in + h * size_w;
      int tmp_h = h * 4;
      for (int w = 0; w < nw; w++) {
        INIT_PTR_4(float, ptr_out, size_h)
#ifdef __aarch64__
        float32x4_t vr0 = vld1q_f32(din0);
        float32x4_t vr1 = vld1q_f32(din1);
        float32x4_t vr2 = vld1q_f32(din2);
        float32x4_t vr3 = vld1q_f32(din3);
        float32x4_t re0 = vtrn1q_f32(vr0, vr1);
        float32x4_t re1 = vtrn2q_f32(vr0, vr1);
        float32x4_t re2 = vtrn1q_f32(vr2, vr3);
        float32x4_t re3 = vtrn2q_f32(vr2, vr3);
        vst1_f32(dout0, vget_low_f32(re0));
        dout0 += 2;
        vst1_f32(dout0, vget_low_f32(re2));
        vst1_f32(dout1, vget_low_f32(re1));
        dout1 += 2;
        vst1_f32(dout1, vget_low_f32(re3));
        vst1_f32(dout2, vget_high_f32(re0));
        dout2 += 2;
        vst1_f32(dout2, vget_high_f32(re2));
        vst1_f32(dout3, vget_high_f32(re1));
        dout3 += 2;
        vst1_f32(dout3, vget_high_f32(re3));
#else
        asm("vld1.32 {d0, d1}, [%[in0]]    \n"
            "vld1.32 {d2, d3}, [%[in1]]    \n"
            "vld1.32 {d4, d5}, [%[in2]]    \n"
            "vld1.32 {d6, d7}, [%[in3]]    \n"
            "vtrn.32 q0, q1                \n"
            "vtrn.32 q2, q3                \n"
            "vswp d1, d4                   \n"
            "vswp d3, d6                   \n"
            "vst1.32 {d0, d1}, [%[out0]]   \n"
            "vst1.32 {d2, d3}, [%[out1]]   \n"
            "vst1.32 {d4, d5}, [%[out2]]   \n"
            "vst1.32 {d6, d7}, [%[out3]]   \n"
            :
            : [out0] "r"(dout0),
              [out1] "r"(dout1),
              [out2] "r"(dout2),
              [out3] "r"(dout3),
              [in0] "r"(din0),
              [in1] "r"(din1),
              [in2] "r"(din2),
              [in3] "r"(din3)
            : "q0", "q1", "q2", "q3");
#endif
        ptr_din_row += 4;
      }
    }
    LITE_PARALLEL_END();
    // remain
    for (int h = 0; h < height; h++) {
      for (int w = nw * 4; w < width; w++) {
        const float* data_in_ptr = ptr_in + h * width + w;
        float* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
    for (int w = 0; w < width; w++) {
      for (int h = nh * 4; h < height; h++) {
        const float* data_in_ptr = ptr_in + h * width + w;
        float* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
  }
}

#ifdef ENABLE_ARM_FP16
void transpose_mat(const lite_api::float16_t* din,
                   lite_api::float16_t* dout,
                   const int num,
                   const int width,
                   const int height) {
  int nw = width >> 3;
#ifdef __aarch64__
  int nh = height >> 3;
  int size_w = width << 3;
#else
  int nh = height >> 2;
  int size_w = width << 2;
#endif
  int size_in = width * height;
  int size_h = height << 3;
  int remain_w = (width & 7);
  int remain_ww = remain_w >> 2;
  int remain_ww_rem = remain_w & 3;
  int size_wh = nw * size_h;
  for (int i = 0; i < num; ++i) {
    lite_api::float16_t* ptr_out = dout + i * size_in;
    const lite_api::float16_t* ptr_in = din + i * size_in;
    LITE_PARALLEL_BEGIN(h, tid, nh) {
      const lite_api::float16_t* ptr_din_row = ptr_in + h * size_w;
#ifdef __aarch64__
      int tmp_h = h << 3;
#else
      int tmp_h = h << 2;
#endif
      for (int w = 0; w < nw; w++) {
        INIT_PTR_4(lite_api::float16_t, ptr_out, size_h)
#ifdef __aarch64__
        INIT_PTR_A4(lite_api::float16_t)
        INIT_PTR_B4(lite_api::float16_t)
        asm volatile(
            "ldr q0, [%[din0]], #16\n"
            "ldr q1, [%[din1]], #16\n"
            "ldr q2, [%[din2]], #16\n"
            "ldr q3, [%[din3]], #16\n"
            "ldr q4, [%[din4]], #16\n"
            "ldr q5, [%[din5]], #16\n"
            // a0b0a2b2a4b4a6b6
            "trn1 v8.8h, v0.8h, v1.8h\n"
            // a1b1a3b3a5b5a7b7
            "trn2 v9.8h, v0.8h, v1.8h\n"
            "ldr q6, [%[din6]], #16\n"
            "trn1 v10.8h, v2.8h, v3.8h\n"
            "trn2 v11.8h, v2.8h, v3.8h\n"
            "ldr q7, [%[din7]], #16\n"
            "trn1 v12.8h, v4.8h, v5.8h\n"
            "trn2 v13.8h, v4.8h, v5.8h\n"

            // a0b0c0d0a4b4c4d4
            "trn1 v0.4s, v8.4s, v10.4s\n"
            // a2b2c2d2a6b6c6d6
            "trn2 v1.4s, v8.4s, v10.4s\n"
            "trn1 v14.8h, v6.8h, v7.8h\n"
            "trn2 v15.8h, v6.8h, v7.8h\n"
            // a1b1c1d1a5b5c5d5
            "trn1 v2.4s, v9.4s, v11.4s\n"
            // a3b3c3d3a7b7c7d7
            "trn2 v3.4s, v9.4s, v11.4s\n"

            // e0f0g0h0a4b4c4d4
            "trn1 v4.4s, v12.4s, v14.4s\n"
            "trn2 v5.4s, v12.4s, v14.4s\n"

            // e1f1g1h1a5b5c5d5
            "trn1 v6.4s, v13.4s, v15.4s\n"
            "trn2 v7.4s, v13.4s, v15.4s\n"

            "trn1 v8.2d, v0.2d, v4.2d\n"   // 0
            "trn2 v9.2d, v0.2d, v4.2d\n"   // 4
            "trn1 v10.2d, v1.2d, v5.2d\n"  // 2
            "trn2 v11.2d, v1.2d, v5.2d\n"  // 6
            "trn1 v12.2d, v2.2d, v6.2d\n"  // 1
            "str q8, [%[dout0]]\n"
            "trn2 v13.2d, v2.2d, v6.2d\n"  // 3
            "str q9, [%[dout4]]\n"
            "trn1 v14.2d, v3.2d, v7.2d\n"  // 5
            "str q10, [%[dout2]]\n"
            "trn2 v15.2d, v3.2d, v7.2d\n"  // 7
            "str q11, [%[dout6]]\n"
            "str q12, [%[dout1]]\n"
            "str q14, [%[dout3]]\n"
            "str q13, [%[dout5]]\n"
            "str q15, [%[dout7]]\n"
            : [din0] "+r"(din0),
              [din1] "+r"(din1),
              [din2] "+r"(din2),
              [din3] "+r"(din3),
              [din4] "+r"(din4),
              [din5] "+r"(din5),
              [din6] "+r"(din6),
              [din7] "+r"(din7)
            : [dout0] "r"(dout0),
              [dout1] "r"(dout1),
              [dout2] "r"(dout2),
              [dout3] "r"(dout3),
              [dout4] "r"(dout4),
              [dout5] "r"(dout5),
              [dout6] "r"(dout6),
              [dout7] "r"(dout7)
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v8",
              "v9",
              "v10",
              "v11",
              "v12",
              "v13",
              "v14",
              "v15");
#else
        INIT_PTR_B4(lite_api::float16_t)
        asm volatile(
            "vld1.16 {d0-d1}, [%[din0]]!\n"
            "vld1.16 {d2-d3}, [%[din1]]!\n"
            "vld1.16 {d4-d5}, [%[din2]]!\n"
            "vld1.16 {d6-d7}, [%[din3]]!\n"
            // q0 =a0b0a2b2a4b4a6b6 q1 = a1b1a3b3a5b5a7b7
            "vtrn.16 q0, q1\n"
            "vtrn.16 q2, q3\n"

            // q0 = a0b0c0d0a4b4c4d4, q2 = a2b2c2d2a6b6c6d6
            "vtrn.32 q0, q2\n"
            "vtrn.32 q1, q3\n"

            "vst1.16 {d0}, [%[dout0]]!\n"
            "vst1.16 {d2}, [%[dout1]]!\n"
            "vst1.16 {d4}, [%[dout2]]!\n"
            "vst1.16 {d6}, [%[dout3]]!\n"
            "vst1.16 {d1}, [%[dout4]]!\n"
            "vst1.16 {d3}, [%[dout5]]!\n"
            "vst1.16 {d5}, [%[dout6]]!\n"
            "vst1.16 {d7}, [%[dout7]]!\n"
            : [din0] "+r"(din0),
              [din1] "+r"(din1),
              [din2] "+r"(din2),
              [din3] "+r"(din3),
              [dout0] "+r"(dout0),
              [dout1] "+r"(dout1),
              [dout2] "+r"(dout2),
              [dout3] "+r"(dout3),
              [dout4] "+r"(dout4),
              [dout5] "+r"(dout5),
              [dout6] "+r"(dout6),
              [dout7] "+r"(dout7)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
        ptr_din_row += 8;
      }
      lite_api::float16_t* data_out_ptr0 = ptr_out + size_wh;
      for (int w = 0; w < remain_ww; w++) {
        INIT_PTR_4(lite_api::float16_t, data_out_ptr0, (4 * height))
#ifdef __aarch64__
        INIT_PTR_A4(lite_api::float16_t)
        asm volatile(
            "ldr d0, [%[din0]], #8\n"
            "ldr d1, [%[din1]], #8\n"
            "ldr d2, [%[din2]], #8\n"
            "ldr d3, [%[din3]], #8\n"
            "ldr d4, [%[din4]], #8\n"
            "ldr d5, [%[din5]], #8\n"
            // a0b0a2b2
            "trn1 v8.4h, v0.4h, v1.4h\n"
            // a1b1a3b3
            "trn2 v9.4h, v0.4h, v1.4h\n"
            "ldr d6, [%[din6]], #8\n"
            "trn1 v10.4h, v2.4h, v3.4h\n"
            "trn2 v11.4h, v2.4h, v3.4h\n"
            "ldr d7, [%[din7]], #8\n"
            "trn1 v12.4h, v4.4h, v5.4h\n"
            "trn2 v13.4h, v4.4h, v5.4h\n"

            // a0b0c0d0
            "trn1 v0.2s, v8.2s, v10.2s\n"
            // a2b2c2d2
            "trn2 v1.2s, v8.2s, v10.2s\n"
            "trn1 v14.4h, v6.4h, v7.4h\n"
            "trn2 v15.4h, v6.4h, v7.4h\n"
            // a1b1c1d1
            "trn1 v2.2s, v9.2s, v11.2s\n"
            // a3b3c3d3
            "trn2 v3.2s, v9.2s, v11.2s\n"

            // e0f0g0h0
            "trn1 v4.2s, v12.2s, v14.2s\n"
            "trn2 v5.2s, v12.2s, v14.2s\n"
            // e1f1g1h1
            "trn1 v6.2s, v13.2s, v15.2s\n"
            "trn2 v7.2s, v13.2s, v15.2s\n"

            "str d0, [%[dout0]], #8\n"
            "str d1, [%[dout2]], #8\n"
            "str d2, [%[dout1]], #8\n"
            "str d3, [%[dout3]], #8\n"
            "str d4, [%[dout0]], #8\n"
            "str d5, [%[dout2]], #8\n"
            "str d6, [%[dout1]], #8\n"
            "str d7, [%[dout3]], #8\n"
            : [din0] "+r"(din0),
              [din1] "+r"(din1),
              [din2] "+r"(din2),
              [din3] "+r"(din3),
              [din4] "+r"(din4),
              [din5] "+r"(din5),
              [din6] "+r"(din6),
              [din7] "+r"(din7),
              [dout0] "+r"(dout0),
              [dout1] "+r"(dout1),
              [dout2] "+r"(dout2),
              [dout3] "+r"(dout3)
            :
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v8",
              "v9",
              "v10",
              "v11",
              "v12",
              "v13",
              "v14",
              "v15");
#else
        asm volatile(
            "vld1.16 {d0}, [%[din0]]!\n"
            "vld1.16 {d2}, [%[din1]]!\n"
            "vld1.16 {d4}, [%[din2]]!\n"
            "vld1.16 {d6}, [%[din3]]!\n"
            // q0 =a0b0a2b2 q1 = a1b1a3b3
            "vtrn.16 d0, d2\n"
            "vtrn.16 d4, d6\n"

            // q0 = a0b0c0d0, q2 = a2b2c2d2
            "vtrn.32 d0, d4\n"
            "vtrn.32 d2, d6\n"

            "vst1.16 {d0}, [%[dout0]]!\n"
            "vst1.16 {d4}, [%[dout2]]!\n"
            "vst1.16 {d2}, [%[dout1]]!\n"
            "vst1.16 {d6}, [%[dout3]]!\n"
            : [din0] "+r"(din0),
              [din1] "+r"(din1),
              [din2] "+r"(din2),
              [din3] "+r"(din3),
              [dout0] "+r"(dout0),
              [dout1] "+r"(dout1),
              [dout2] "+r"(dout2),
              [dout3] "+r"(dout3)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
        ptr_din_row += 4;
      }
      data_out_ptr0 = ptr_out + size_wh + 4 * height * remain_ww;
      for (int w = 0; w < remain_ww_rem; w++) {
        INIT_PTR_4(lite_api::float16_t, data_out_ptr0, height)
        INIT_PTR_A4(lite_api::float16_t)
        *data_out_ptr++ = *din0;
        *data_out_ptr++ = *din1;
        *data_out_ptr++ = *din2;
        *data_out_ptr++ = *din3;
        *data_out_ptr++ = *din4;
        *data_out_ptr++ = *din5;
        *data_out_ptr++ = *din6;
        *data_out_ptr++ = *din7;
        ptr_din_row++;
      }
    }
    LITE_PARALLEL_END()
#ifdef __aarch64__
    // remain
    for (int h = nh * 8; h < height; h++) {
#else
    for (int h = nh * 4; h < height; h++) {
#endif
      for (int w = 0; w < width; w++) {
        const float16_t* data_in_ptr = ptr_in + h * width + w;
        float16_t* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
  }
}
#endif

std::vector<int> get_stride(const paddle::lite::DDimLite& dims) {
  std::vector<int> data_stride{0};

  for (int i = 0; i < dims.size(); ++i) {
    data_stride.push_back(dims.count(i, dims.size()));
  }
  return data_stride;
}

void TransposeCompute::ReInitWhenNeeded() {
  auto& param = Param<operators::TransposeParam>();
  auto* input = param.x;
  auto* output = param.output;

  auto x_dims = param.x->dims();
  if (last_shape_ == x_dims) {
    return;
  }
  last_shape_ = x_dims;
  int _num_axes = input->dims().size();
  CHECK(_num_axes == param.axis.size())
      << "axis size is not match to input dims";

  need_trans = false;
  for (int i = 0; i < _num_axes; ++i) {
    if (param.axis[i] != i) {
      need_trans = true;
      break;
    }
  }

  if (!need_trans) {
    return;
  }

  std::vector<int> axis_diff;
  int j = 0;
  for (int i = 0; i < _num_axes; ++i) {
    if (param.axis[j] != i) {
      axis_diff.push_back(j);
    } else {
      j++;
    }
  }
  for (int i = 0; i < axis_diff.size(); i++) {
  }
  if (input->dims().count(axis_diff[0], _num_axes) == 1) {
    need_trans = false;
    return;
  }

  if (axis_diff.size() == 1) {
    trans_mat = true;
    _trans_num = input->dims().count(0, std::max(axis_diff[0], 0));
    _trans_w = input->dims().count(axis_diff[0] + 1, _num_axes);
    _trans_h = input->dims()[axis_diff[0]];
  } else {
    trans_mat = false;
    _new_steps = get_stride(output->dims());
    _old_steps = get_stride(input->dims());
  }
}
void TransposeCompute::PrepareForRun() { ReInitWhenNeeded(); }

template <typename Dtype>
void TransposeCompute_(const std::vector<int>& axis,
                       const lite::Tensor* input,
                       lite::Tensor* output) {
  const Dtype* input_ptr = input->data<Dtype>();
  Dtype* output_ptr = output->mutable_data<Dtype>();

  // input and output's shape dimension must >= 2 && <= 6.
  const DDim& in_dim = input->dims();
  const DDim& out_dim = output->dims();

  // precompute inverted output dim and strides
  size_t rout_dim[6], strides[6];
  int permute = axis.size();  // permute must >=2 && <= 6.
  for (int i = 0; i < permute; ++i) {
    int k = permute - 1 - i;
    strides[k] = 1;
    for (int j = axis[i] + 1; j < permute; ++j) {
      strides[k] *= in_dim[j];
    }
    rout_dim[k] = out_dim[i];
  }

  // unroll the first 2 dimensions
  int reamin_dim = 1;
  for (int i = 2; i < out_dim.size(); ++i) {
    reamin_dim *= out_dim[i];
  }

  for (int batch = 0; batch < out_dim[0]; ++batch) {
    LITE_PARALLEL_BEGIN(j, tid, out_dim[1]) {
      size_t offset = batch * strides[permute - 1] + j * strides[permute - 2];
      Dtype* out_ptr = output_ptr + (batch * out_dim[1] + j) * reamin_dim;
      int indics[4] = {0, 0, 0, 0};
      for (int k = 0; k < reamin_dim; ++k) {
        out_ptr[k] = input_ptr[offset];
        indics[0] += 1;
        offset += strides[0];
        for (int p = 0; p < permute - 3; ++p) {
          if (indics[p] == rout_dim[p]) {
            indics[p + 1] += 1;
            indics[p] = 0;
            offset += strides[p + 1];
            offset -= rout_dim[p] * strides[p];
          } else {
            break;
          }
        }
      }
    }
    LITE_PARALLEL_END();
  }
}
// Transpose
void TransposeCompute::Run() {
  auto& param = Param<operators::TransposeParam>();
  auto* input = param.x;
  auto* output = param.output;
  const std::vector<int> axis = param.axis;
  //! only copy the data
  if (!need_trans) {
    output->CopyDataFrom(*input);
    return;
  }

  if (input->precision() == PRECISION(kFloat) && trans_mat) {
    const float* din = input->data<float>();
    float* dout = output->mutable_data<float>();
    transpose_mat(din, dout, _trans_num, _trans_w, _trans_h);
    return;
  }
#ifdef ENABLE_ARM_FP16
  if (input->precision() == PRECISION(kFP16) && trans_mat) {
    const lite_api::float16_t* din = input->data<lite_api::float16_t>();
    lite_api::float16_t* dout = output->mutable_data<lite_api::float16_t>();
    transpose_mat(din, dout, _trans_num, _trans_w, _trans_h);
    return;
  }
#endif

  switch (input->precision()) {
    case PRECISION(kInt8):
      TransposeCompute_<int8_t>(axis, input, output);
      break;
    case PRECISION(kInt32):
      TransposeCompute_<int32_t>(axis, input, output);
      break;
    case PRECISION(kInt64):
      TransposeCompute_<int64_t>(axis, input, output);
      break;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      TransposeCompute_<lite_api::float16_t>(axis, input, output);
      break;
#endif
    case PRECISION(kFloat):
      TransposeCompute_<float>(axis, input, output);
      break;
    case PRECISION(kFP64):
      TransposeCompute_<double>(axis, input, output);
      break;
    default:
      LOG(FATAL) << "Not support the dtype: "
                 << static_cast<int>(input->precision());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// Transpose
REGISTER_LITE_KERNEL(transpose,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::TransposeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();

// Transpose2
REGISTER_LITE_KERNEL(transpose2,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::Transpose2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
