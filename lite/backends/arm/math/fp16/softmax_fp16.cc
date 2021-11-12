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
#include "lite/backends/arm/math/fp16/softmax_fp16.h"
#include <algorithm>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void softmax_basic_fp16(const float16_t* din,
                        float16_t* dout,
                        const int axis_size,
                        const int inner_num,
                        const int outer_num) {
  int compute_size = inner_num * outer_num;

  LITE_PARALLEL_BEGIN(i, tid, compute_size) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    float16_t max_data = din[real_index];
    // get max
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      max_data = din[real_index] > max_data ? din[real_index] : max_data;
    }

    real_index = idx_outer * inner_num + idx_inner;
    // sub, exp and sum
    dout[real_index] = expf(din[real_index] - max_data);
    float16_t sum_data = dout[real_index];
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      dout[real_index] = expf(din[real_index] - max_data);
      sum_data += dout[real_index];
    }

    float16_t sum_inv = 1.f / sum_data;
    real_index = idx_outer * inner_num + idx_inner;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      dout[real_index] *= sum_inv;
      real_index += inner_num;
    }
  }
  LITE_PARALLEL_END()
}

void softmax_inner8_axis4_fp16(const float16_t* din,
                               float16_t* dout,
                               const int axis_size,
                               const int inner_num,
                               const int outer_num) {
  int compute_size = inner_num * outer_num;
  int cmp_cnt = compute_size >> 3;
  int remain = compute_size % 8;
  float16x8_t vone = vdupq_n_f16(1.0f);

  LITE_PARALLEL_BEGIN(c, tid, cmp_cnt) {
    int i = c * 8;
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    // get max axis_size == 4
    const float16_t* din_ptr = din + real_index;
    const float16_t* din_ptr1 = din_ptr + inner_num;
    const float16_t* din_ptr2 = din_ptr1 + inner_num;
    const float16_t* din_ptr3 = din_ptr2 + inner_num;
    float16x8_t vdata0 = vld1q_f16(din_ptr);
    float16x8_t vdata1 = vld1q_f16(din_ptr1);
    float16x8_t vdata2 = vld1q_f16(din_ptr2);
    float16x8_t vdata3 = vld1q_f16(din_ptr3);
    float16_t* dout_ptr0 = dout + real_index;
    float16_t* dout_ptr1 = dout_ptr0 + inner_num;

    float16x8_t vmax1 = vmaxq_f16(vdata0, vdata1);
    float16x8_t vmax2 = vmaxq_f16(vdata2, vdata3);
    float16_t* dout_ptr2 = dout_ptr1 + inner_num;
    float16_t* dout_ptr3 = dout_ptr2 + inner_num;
    float16x8_t vmax = vmaxq_f16(vmax1, vmax2);

    // sub, exp and sum
    float16x8_t vsum0 = expq_ps_f16(vsubq_f16(vdata0, vmax));
    float16x8_t vsum1 = expq_ps_f16(vsubq_f16(vdata1, vmax));
    float16x8_t vsum2 = expq_ps_f16(vsubq_f16(vdata2, vmax));
    float16x8_t vsum3 = expq_ps_f16(vsubq_f16(vdata3, vmax));

    float16x8_t vsum_1 = vaddq_f16(vsum0, vsum1);
    float16x8_t vsum_2 = vaddq_f16(vsum2, vsum3);

    float16x8_t vsum = vaddq_f16(vsum_1, vsum_2);

    float16x8_t vinf = divq_ps_f16(vone, vsum);

    vsum0 = vmulq_f16(vsum0, vinf);
    vsum1 = vmulq_f16(vsum1, vinf);
    vsum2 = vmulq_f16(vsum2, vinf);
    vsum3 = vmulq_f16(vsum3, vinf);

    vst1q_f16(dout_ptr0, vsum0);
    vst1q_f16(dout_ptr1, vsum1);
    vst1q_f16(dout_ptr2, vsum2);
    vst1q_f16(dout_ptr3, vsum3);
  }
  LITE_PARALLEL_END()

  int i = cmp_cnt * 8;

  if (remain >= 4) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;
    // get max axis_size == 4
    const float16_t* din_ptr = din + real_index;
    const float16_t* din_ptr1 = din_ptr + inner_num;
    const float16_t* din_ptr2 = din_ptr1 + inner_num;
    const float16_t* din_ptr3 = din_ptr2 + inner_num;
    float16x4_t vdata0 = vld1_f16(din_ptr);
    float16x4_t vdata1 = vld1_f16(din_ptr1);
    float16x4_t vdata2 = vld1_f16(din_ptr2);
    float16x4_t vdata3 = vld1_f16(din_ptr3);

    float16_t* dout_ptr0 = dout + real_index;
    float16_t* dout_ptr1 = dout_ptr0 + inner_num;
    float16x4_t vmax1 = vmax_f16(vdata0, vdata1);
    float16x4_t vmax2 = vmax_f16(vdata2, vdata3);
    float16_t* dout_ptr2 = dout_ptr1 + inner_num;
    float16_t* dout_ptr3 = dout_ptr2 + inner_num;
    float16x4_t vmax = vmax_f16(vmax1, vmax2);

    // sub, exp and sum
    float16x4_t vsum0 = exp_ps_f16(vsub_f16(vdata0, vmax));
    float16x4_t vsum1 = exp_ps_f16(vsub_f16(vdata1, vmax));
    float16x4_t vsum2 = exp_ps_f16(vsub_f16(vdata2, vmax));
    float16x4_t vsum3 = exp_ps_f16(vsub_f16(vdata3, vmax));

    float16x4_t vsum_1 = vadd_f16(vsum0, vsum1);
    float16x4_t vsum_2 = vadd_f16(vsum2, vsum3);

    float16x4_t vsum = vadd_f16(vsum_1, vsum_2);

    float16x4_t vone = vdup_n_f16(1.0f);
    float16x4_t vinf = div_ps_f16(vone, vsum);

    vsum0 = vmul_f16(vsum0, vinf);
    vsum1 = vmul_f16(vsum1, vinf);
    vsum2 = vmul_f16(vsum2, vinf);
    vsum3 = vmul_f16(vsum3, vinf);

    vst1_f16(dout_ptr0, vsum0);
    vst1_f16(dout_ptr1, vsum1);
    vst1_f16(dout_ptr2, vsum2);
    vst1_f16(dout_ptr3, vsum3);

    i += 4;
  }

  for (; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    float max_data = din[real_index];
    // get max
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      max_data = din[real_index] > max_data ? din[real_index] : max_data;
    }

    real_index = idx_outer * inner_num + idx_inner;
    // sub, exp and sum
    dout[real_index] = expf(din[real_index] - max_data);
    float16_t sum_data = dout[real_index];
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      dout[real_index] = expf(din[real_index] - max_data);
      sum_data += dout[real_index];
    }

    real_index = idx_outer * inner_num + idx_inner;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      dout[real_index] /= sum_data;
      real_index += inner_num;
    }
  }
}

void softmax_inner8_axis1_fp16(const float16_t* din,
                               float16_t* dout,
                               const int axis_size,
                               const int inner_num,
                               const int outer_num) {
  int compute_size = inner_num * outer_num;
  int cmp_cnt = compute_size >> 3;
  int remain = compute_size & 7;

  float16x8_t vone = vdupq_n_f16(1.0f);

  LITE_PARALLEL_BEGIN(c, tid, cmp_cnt) {
    int i = c * 8;
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    const float16_t* din_ptr = din + real_index;
    float16x8_t vmax = vld1q_f16(din_ptr);
    // get max
    for (int j = 1; j < axis_size; ++j) {
      din_ptr += inner_num;
      float16x8_t vdata = vld1q_f16(din_ptr);
      vmax = vmaxq_f16(vmax, vdata);
    }

    // sub, exp and sum
    din_ptr = din + real_index;
    float16_t* dout_ptr = dout + real_index;
    float16x8_t vdata = vld1q_f16(din_ptr);
    float16x8_t vsum = expq_ps_f16(vsubq_f16(vdata, vmax));
    din_ptr += inner_num;
    vst1q_f16(dout_ptr, vsum);
    dout_ptr += inner_num;
    for (int j = 1; j < axis_size; ++j) {
      float16x8_t vdata0 = vld1q_f16(din_ptr);
      vdata0 = expq_ps_f16(vsubq_f16(vdata0, vmax));
      din_ptr += inner_num;
      vsum = vaddq_f16(vsum, vdata0);
      vst1q_f16(dout_ptr, vdata0);
      dout_ptr += inner_num;
    }

    float16x8_t vinf = divq_ps_f16(vone, vsum);
    dout_ptr = dout + real_index;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      float16x8_t vdata0 = vld1q_f16(dout_ptr);
      vdata0 = vmulq_f16(vdata0, vinf);
      vst1q_f16(dout_ptr, vdata0);
      dout_ptr += inner_num;
    }
  }
  LITE_PARALLEL_END()
  int index = cmp_cnt << 3;
  if (remain >= 4) {
    int idx_inner = index % inner_num;
    int idx_outer = (index / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    const float16_t* din_ptr = din + real_index;
    float16x4_t vmax = vld1_f16(din_ptr);
    // get max
    for (int j = 1; j < axis_size; ++j) {
      din_ptr += inner_num;
      float16x4_t vdata = vld1_f16(din_ptr);
      vmax = vmax_f16(vmax, vdata);
    }

    // sub, exp and sum
    din_ptr = din + real_index;
    float16_t* dout_ptr = dout + real_index;
    float16x4_t vdata = vld1_f16(din_ptr);
    float16x4_t vsum = exp_ps_f16(vsub_f16(vdata, vmax));
    din_ptr += inner_num;
    vst1_f16(dout_ptr, vsum);
    dout_ptr += inner_num;
    for (int j = 1; j < axis_size; ++j) {
      float16x4_t vdata0 = vld1_f16(din_ptr);
      vdata0 = exp_ps_f16(vsub_f16(vdata0, vmax));
      din_ptr += inner_num;
      vsum = vadd_f16(vsum, vdata0);
      vst1_f16(dout_ptr, vdata0);
      dout_ptr += inner_num;
    }
    float16x4_t vinf = div_ps_f16(vget_high_f16(vone), vsum);
    dout_ptr = dout + real_index;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      float16x4_t vdata0 = vld1_f16(dout_ptr);
      vdata0 = vmul_f16(vdata0, vinf);
      vst1_f16(dout_ptr, vdata0);
      dout_ptr += inner_num;
    }
    index += 4;
  }

  for (int i = index; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    float16_t max_data = din[real_index];
    // get max
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      max_data = din[real_index] > max_data ? din[real_index] : max_data;
    }

    real_index = idx_outer * inner_num + idx_inner;
    // sub, exp and sum
    dout[real_index] = expf(din[real_index] - max_data);
    float16_t sum_data = dout[real_index];
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      dout[real_index] = expf(din[real_index] - max_data);
      sum_data += dout[real_index];
    }

    real_index = idx_outer * inner_num + idx_inner;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      dout[real_index] /= sum_data;
      real_index += inner_num;
    }
  }
}

void softmax_inner1_large_axis_fp16(const float16_t* din,
                                    float16_t* dout,
                                    const int outer_size,
                                    const int axis_size) {
  int cmp_cnt = axis_size >> 3;
  int remain = axis_size & 7;
  int out_cnt = (outer_size >> 2) << 2;

  LITE_PARALLEL_COMMON_BEGIN(i, tid, outer_size - 3, 0, 4) {
    const float16_t* din_ptr0 = din + i * axis_size;
    float16_t* dout_ptr0 = dout + i * axis_size;
    const float16_t* din_ptr1 = din_ptr0 + axis_size;
    float16_t* dout_ptr1 = dout_ptr0 + axis_size;
    const float16_t* din_ptr2 = din_ptr1 + axis_size;
    float16_t* dout_ptr2 = dout_ptr1 + axis_size;
    const float16_t* din_ptr3 = din_ptr2 + axis_size;
    float16_t* dout_ptr3 = dout_ptr2 + axis_size;

    const float16_t* din_max_ptr0 = din_ptr0;
    const float16_t* din_max_ptr1 = din_ptr1;
    const float16_t* din_max_ptr2 = din_ptr2;
    const float16_t* din_max_ptr3 = din_ptr3;

    // get max
    float16x8_t vmax0 = vld1q_f16(din_max_ptr0);
    float16x8_t vmax1 = vld1q_f16(din_max_ptr1);
    float16x8_t vmax2 = vld1q_f16(din_max_ptr2);
    float16x8_t vmax3 = vld1q_f16(din_max_ptr3);
    din_max_ptr0 += 8;
    din_max_ptr1 += 8;
    din_max_ptr2 += 8;
    din_max_ptr3 += 8;
    for (int j = 1; j < cmp_cnt; j++) {
      vmax0 = vmaxq_f16(vmax0, vld1q_f16(din_max_ptr0));
      vmax1 = vmaxq_f16(vmax1, vld1q_f16(din_max_ptr1));
      vmax2 = vmaxq_f16(vmax2, vld1q_f16(din_max_ptr2));
      vmax3 = vmaxq_f16(vmax3, vld1q_f16(din_max_ptr3));
      din_max_ptr0 += 8;
      din_max_ptr1 += 8;
      din_max_ptr2 += 8;
      din_max_ptr3 += 8;
    }
    float16x4_t vhmax0 = vmax_f16(vget_high_f16(vmax0), vget_low_f16(vmax0));
    float16x4_t vhmax1 = vmax_f16(vget_high_f16(vmax1), vget_low_f16(vmax1));
    float16x4_t vhmax2 = vmax_f16(vget_high_f16(vmax2), vget_low_f16(vmax2));
    float16x4_t vhmax3 = vmax_f16(vget_high_f16(vmax3), vget_low_f16(vmax3));
    int index = cmp_cnt << 3;
    if (remain >= 4) {
      vhmax0 = vmax_f16(vhmax0, vld1_f16(din_max_ptr0));
      vhmax1 = vmax_f16(vhmax1, vld1_f16(din_max_ptr1));
      vhmax2 = vmax_f16(vhmax2, vld1_f16(din_max_ptr2));
      vhmax3 = vmax_f16(vhmax3, vld1_f16(din_max_ptr3));
      din_max_ptr0 += 4;
      din_max_ptr1 += 4;
      din_max_ptr2 += 4;
      din_max_ptr3 += 4;
      index += 4;
    }
    float16x4_t vhpmax0 = vpmax_f16(vhmax0, vhmax0);
    float16x4_t vhpmax1 = vpmax_f16(vhmax1, vhmax1);
    float16x4_t vhpmax2 = vpmax_f16(vhmax2, vhmax2);
    float16x4_t vhpmax3 = vpmax_f16(vhmax3, vhmax3);
    float16x4_t vhlmax0 = vpmax_f16(vhpmax0, vhpmax0);
    float16x4_t vhlmax1 = vpmax_f16(vhpmax1, vhpmax1);
    float16x4_t vhlmax2 = vpmax_f16(vhpmax2, vhpmax2);
    float16x4_t vhlmax3 = vpmax_f16(vhpmax3, vhpmax3);
    for (int j = index; j < axis_size; j++) {
      vhlmax0[0] = std::max(vhlmax0[0], *din_max_ptr0);
      vhlmax1[0] = std::max(vhlmax1[0], *din_max_ptr1);
      vhlmax2[0] = std::max(vhlmax2[0], *din_max_ptr2);
      vhlmax3[0] = std::max(vhlmax3[0], *din_max_ptr3);
      din_max_ptr0++;
      din_max_ptr1++;
      din_max_ptr2++;
      din_max_ptr3++;
    }
    vmax0 = vdupq_n_f16(vhlmax0[0]);
    vmax1 = vdupq_n_f16(vhlmax1[0]);
    vmax2 = vdupq_n_f16(vhlmax2[0]);
    vmax3 = vdupq_n_f16(vhlmax3[0]);
    // sub, exp and sum
    const float16_t* din_sum_ptr0 = din_ptr0;
    float16_t* dout_sum_ptr0 = dout_ptr0;
    const float16_t* din_sum_ptr1 = din_ptr1;
    float16_t* dout_sum_ptr1 = dout_ptr1;
    const float16_t* din_sum_ptr2 = din_ptr2;
    float16_t* dout_sum_ptr2 = dout_ptr2;
    const float16_t* din_sum_ptr3 = din_ptr3;
    float16_t* dout_sum_ptr3 = dout_ptr3;
    float16x8_t vsum0 = vdupq_n_f16(0.f);
    float16x8_t vsum1 = vdupq_n_f16(0.f);
    float16x8_t vsum2 = vdupq_n_f16(0.f);
    float16x8_t vsum3 = vdupq_n_f16(0.f);

    for (int j = 0; j < cmp_cnt; j++) {
      float16x8_t vsub_exp0 =
          expq_ps_f16(vsubq_f16(vld1q_f16(din_sum_ptr0), vmax0));
      float16x8_t vsub_exp1 =
          expq_ps_f16(vsubq_f16(vld1q_f16(din_sum_ptr1), vmax1));
      float16x8_t vsub_exp2 =
          expq_ps_f16(vsubq_f16(vld1q_f16(din_sum_ptr2), vmax2));
      float16x8_t vsub_exp3 =
          expq_ps_f16(vsubq_f16(vld1q_f16(din_sum_ptr3), vmax3));
      vsum0 = vaddq_f16(vsum0, vsub_exp0);
      vsum1 = vaddq_f16(vsum1, vsub_exp1);
      vsum2 = vaddq_f16(vsum2, vsub_exp2);
      vsum3 = vaddq_f16(vsum3, vsub_exp3);
      vst1q_f16(dout_sum_ptr0, vsub_exp0);
      din_sum_ptr0 += 8;
      vst1q_f16(dout_sum_ptr1, vsub_exp1);
      din_sum_ptr1 += 8;
      vst1q_f16(dout_sum_ptr2, vsub_exp2);
      din_sum_ptr2 += 8;
      vst1q_f16(dout_sum_ptr3, vsub_exp3);
      din_sum_ptr3 += 8;
      dout_sum_ptr0 += 8;
      dout_sum_ptr1 += 8;
      dout_sum_ptr2 += 8;
      dout_sum_ptr3 += 8;
    }
    float16x4_t vhsum0 = vadd_f16(vget_high_f16(vsum0), vget_low_f16(vsum0));
    float16x4_t vhsum1 = vadd_f16(vget_high_f16(vsum1), vget_low_f16(vsum1));
    float16x4_t vhsum2 = vadd_f16(vget_high_f16(vsum2), vget_low_f16(vsum2));
    float16x4_t vhsum3 = vadd_f16(vget_high_f16(vsum3), vget_low_f16(vsum3));
    index = cmp_cnt << 3;
    if (remain >= 4) {
      float16x4_t vsub_exp0 =
          exp_ps_f16(vsub_f16(vld1_f16(din_sum_ptr0), vget_high_f16(vmax0)));
      float16x4_t vsub_exp1 =
          exp_ps_f16(vsub_f16(vld1_f16(din_sum_ptr1), vget_high_f16(vmax1)));
      float16x4_t vsub_exp2 =
          exp_ps_f16(vsub_f16(vld1_f16(din_sum_ptr2), vget_high_f16(vmax2)));
      float16x4_t vsub_exp3 =
          exp_ps_f16(vsub_f16(vld1_f16(din_sum_ptr3), vget_high_f16(vmax3)));
      vhsum0 = vadd_f16(vhsum0, vsub_exp0);
      vhsum1 = vadd_f16(vhsum1, vsub_exp1);
      vhsum2 = vadd_f16(vhsum2, vsub_exp2);
      vhsum3 = vadd_f16(vhsum3, vsub_exp3);
      vst1_f16(dout_sum_ptr0, vsub_exp0);
      din_sum_ptr0 += 4;
      vst1_f16(dout_sum_ptr1, vsub_exp1);
      din_sum_ptr1 += 4;
      vst1_f16(dout_sum_ptr2, vsub_exp2);
      din_sum_ptr2 += 4;
      vst1_f16(dout_sum_ptr3, vsub_exp3);
      din_sum_ptr3 += 4;
      dout_sum_ptr0 += 4;
      dout_sum_ptr1 += 4;
      dout_sum_ptr2 += 4;
      dout_sum_ptr3 += 4;
      index += 4;
    }
    float16_t sum_data0 = vhsum0[0] + vhsum0[1] + vhsum0[2] + vhsum0[3];
    float16_t sum_data1 = vhsum1[0] + vhsum1[1] + vhsum1[2] + vhsum1[3];
    float16_t sum_data2 = vhsum2[0] + vhsum2[1] + vhsum2[2] + vhsum2[3];
    float16_t sum_data3 = vhsum3[0] + vhsum3[1] + vhsum3[2] + vhsum3[3];

    for (int j = index; j < axis_size; j++) {
      dout_sum_ptr0[0] = expf(din_sum_ptr0[0] - vmax0[0]);
      dout_sum_ptr1[0] = expf(din_sum_ptr1[0] - vmax1[0]);
      dout_sum_ptr2[0] = expf(din_sum_ptr2[0] - vmax2[0]);
      dout_sum_ptr3[0] = expf(din_sum_ptr3[0] - vmax3[0]);
      sum_data0 += dout_sum_ptr0[0];
      din_sum_ptr0++;
      sum_data1 += dout_sum_ptr1[0];
      din_sum_ptr1++;
      sum_data2 += dout_sum_ptr2[0];
      din_sum_ptr2++;
      sum_data3 += dout_sum_ptr3[0];
      din_sum_ptr3++;
      dout_sum_ptr0++;
      dout_sum_ptr1++;
      dout_sum_ptr2++;
      dout_sum_ptr3++;
    }

    float16_t* dout_res_ptr0 = dout_ptr0;
    float16_t* dout_res_ptr1 = dout_ptr1;
    float16_t* dout_res_ptr2 = dout_ptr2;
    float16_t* dout_res_ptr3 = dout_ptr3;
    float16x8_t vsum_data0 = vdupq_n_f16(sum_data0);
    float16x8_t vsum_data1 = vdupq_n_f16(sum_data1);
    float16x8_t vsum_data2 = vdupq_n_f16(sum_data2);
    float16x8_t vsum_data3 = vdupq_n_f16(sum_data3);
    float16x8_t vinv0 = vrecpeq_f16(vsum_data0);
    float16x8_t vinv1 = vrecpeq_f16(vsum_data1);
    float16x8_t vinv2 = vrecpeq_f16(vsum_data2);
    float16x8_t vinv3 = vrecpeq_f16(vsum_data3);
    // get softmax result
    for (int j = 0; j < cmp_cnt; j++) {
      float16x8_t vout0 = vld1q_f16(dout_res_ptr0);
      float16x8_t vout1 = vld1q_f16(dout_res_ptr1);
      float16x8_t vout2 = vld1q_f16(dout_res_ptr2);
      float16x8_t vout3 = vld1q_f16(dout_res_ptr3);
      float16x8_t vres0 = vmulq_f16(vout0, vinv0);
      float16x8_t vres1 = vmulq_f16(vout1, vinv1);
      float16x8_t vres2 = vmulq_f16(vout2, vinv2);
      float16x8_t vres3 = vmulq_f16(vout3, vinv3);
      vst1q_f16(dout_res_ptr0, vres0);
      vst1q_f16(dout_res_ptr1, vres1);
      vst1q_f16(dout_res_ptr2, vres2);
      vst1q_f16(dout_res_ptr3, vres3);
      dout_res_ptr0 += 8;
      dout_res_ptr1 += 8;
      dout_res_ptr2 += 8;
      dout_res_ptr3 += 8;
    }
    index = cmp_cnt << 3;
    if (remain >= 4) {
      float16x4_t vout0 = vld1_f16(dout_res_ptr0);
      float16x4_t vout1 = vld1_f16(dout_res_ptr1);
      float16x4_t vout2 = vld1_f16(dout_res_ptr2);
      float16x4_t vout3 = vld1_f16(dout_res_ptr3);
      float16x4_t vres0 = vmul_f16(vout0, vget_high_f16(vinv0));
      float16x4_t vres1 = vmul_f16(vout1, vget_high_f16(vinv1));
      float16x4_t vres2 = vmul_f16(vout2, vget_high_f16(vinv2));
      float16x4_t vres3 = vmul_f16(vout3, vget_high_f16(vinv3));
      vst1_f16(dout_res_ptr0, vres0);
      vst1_f16(dout_res_ptr1, vres1);
      vst1_f16(dout_res_ptr2, vres2);
      vst1_f16(dout_res_ptr3, vres3);
      dout_res_ptr0 += 4;
      dout_res_ptr1 += 4;
      dout_res_ptr2 += 4;
      dout_res_ptr3 += 4;
      index += 4;
    }
    for (int j = index; j < axis_size; j++) {
      dout_res_ptr0[0] /= sum_data0;
      dout_res_ptr1[0] /= sum_data1;
      dout_res_ptr2[0] /= sum_data2;
      dout_res_ptr3[0] /= sum_data3;
      dout_res_ptr0++;
      dout_res_ptr1++;
      dout_res_ptr2++;
      dout_res_ptr3++;
    }
  }
  LITE_PARALLEL_END()

  for (int i = out_cnt; i < outer_size; i++) {
    const float16_t* din_ptr0 = din + i * axis_size;
    float16_t* dout_ptr0 = dout + i * axis_size;

    const float16_t* din_max_ptr0 = din_ptr0;

    // get max
    float16x8_t vmax0 = vld1q_f16(din_max_ptr0);
    din_max_ptr0 += 8;
    for (int j = 1; j < cmp_cnt; j++) {
      vmax0 = vmaxq_f16(vmax0, vld1q_f16(din_max_ptr0));
      din_max_ptr0 += 8;
    }
    float16x4_t vhmax0 = vmax_f16(vget_high_f16(vmax0), vget_low_f16(vmax0));
    int index = cmp_cnt << 3;
    if (remain >= 4) {
      vhmax0 = vmax_f16(vhmax0, vld1_f16(din_max_ptr0));
      din_max_ptr0 += 4;
      index += 4;
    }
    float16x4_t vhpmax0 = vpmax_f16(vhmax0, vhmax0);
    float16x4_t vhlmax0 = vpmax_f16(vhpmax0, vhpmax0);
    for (int j = index; j < axis_size; j++) {
      vhlmax0[0] = std::max(vhlmax0[0], *din_max_ptr0);
      din_max_ptr0++;
    }
    vmax0 = vdupq_n_f16(vhlmax0[0]);
    // sub, exp and sum
    const float16_t* din_sum_ptr0 = din_ptr0;
    float16_t* dout_sum_ptr0 = dout_ptr0;
    float16x8_t vsum0 = vdupq_n_f16(0.f);

    for (int j = 0; j < cmp_cnt; j++) {
      float16x8_t vsub_exp0 =
          expq_ps_f16(vsubq_f16(vld1q_f16(din_sum_ptr0), vmax0));
      vsum0 = vaddq_f16(vsum0, vsub_exp0);
      vst1q_f16(dout_sum_ptr0, vsub_exp0);
      din_sum_ptr0 += 8;
      dout_sum_ptr0 += 8;
    }
    float16x4_t vhsum0 = vadd_f16(vget_high_f16(vsum0), vget_low_f16(vsum0));
    index = cmp_cnt << 3;
    if (remain >= 4) {
      float16x4_t vsub_exp0 =
          exp_ps_f16(vsub_f16(vld1_f16(din_sum_ptr0), vget_high_f16(vmax0)));
      vhsum0 = vadd_f16(vhsum0, vsub_exp0);
      vst1_f16(dout_sum_ptr0, vsub_exp0);
      din_sum_ptr0 += 4;
      dout_sum_ptr0 += 4;
      index += 4;
    }
    float16_t sum_data0 = vhsum0[0] + vhsum0[1] + vhsum0[2] + vhsum0[3];

    for (int j = index; j < axis_size; j++) {
      dout_sum_ptr0[0] = expf(din_sum_ptr0[0] - vmax0[0]);
      sum_data0 += dout_sum_ptr0[0];
      din_sum_ptr0++;
      dout_sum_ptr0++;
    }

    float16_t* dout_res_ptr0 = dout_ptr0;
    float16x8_t vsum_data0 = vdupq_n_f16(sum_data0);
    float16x8_t vinv0 = vrecpeq_f16(vsum_data0);
    // get softmax result
    for (int j = 0; j < cmp_cnt; j++) {
      float16x8_t vout0 = vld1q_f16(dout_res_ptr0);
      float16x8_t vres0 = vmulq_f16(vout0, vinv0);
      vst1q_f16(dout_res_ptr0, vres0);
      dout_res_ptr0 += 8;
    }
    index = cmp_cnt << 3;
    if (remain >= 4) {
      float16x4_t vout0 = vld1_f16(dout_res_ptr0);
      float16x4_t vres0 = vmul_f16(vout0, vget_high_f16(vinv0));
      vst1_f16(dout_res_ptr0, vres0);
      dout_res_ptr0 += 4;
      index += 4;
    }
    for (int j = index; j < axis_size; j++) {
      dout_res_ptr0[0] /= sum_data0;
      dout_res_ptr0++;
    }
  }
}

void softmax_inner1_small_axis_fp16(const float16_t* din,
                                    float16_t* dout,
                                    const int outer_size,
                                    const int axis_size) {
  LITE_PARALLEL_BEGIN(i, tid, outer_size) {
    const float16_t* din_ptr = din + i * axis_size;
    float16_t* dout_ptr = dout + i * axis_size;
    // get max
    float16_t max_data = din_ptr[0];
    for (int j = 1; j < axis_size; ++j) {
      max_data = std::max(max_data, din_ptr[j]);
    }

    // sub, exp and sum
    float16_t sum_data = 0.f;
    for (int j = 0; j < axis_size; ++j) {
      dout_ptr[j] = expf(din_ptr[j] - max_data);
      sum_data += dout_ptr[j];
    }

    // float16_t sum_inv = 1.f / sum_data;
    for (int j = 0; j < axis_size; ++j) {
      dout_ptr[j] /= sum_data;
    }
  }
  LITE_PARALLEL_END()
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
