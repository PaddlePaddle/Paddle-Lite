// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/backends/arm/math/sve/softmax_sve.h"
#include <algorithm>
#include "lite/backends/arm/math/sve/funcs_sve.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve {

template <typename Dtype>
void softmax_basic_sve(const Dtype* din,
                       Dtype* dout,
                       const int axis_size,
                       const int inner_num,
                       const int outer_num) {
  int compute_size = inner_num * outer_num;

  LITE_PARALLEL_BEGIN(i, tid, compute_size) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;

    Dtype max_data = din[real_index];
    // get max
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      max_data = din[real_index] > max_data ? din[real_index] : max_data;
    }

    real_index = idx_outer * inner_num + idx_inner;
    // sub, exp and sum
    dout[real_index] = expf(din[real_index] - max_data);
    Dtype sum_data = dout[real_index];
    for (int j = 1; j < axis_size; ++j) {
      real_index += inner_num;
      dout[real_index] = expf(din[real_index] - max_data);
      sum_data += dout[real_index];
    }

    Dtype sum_inv = 1.f / sum_data;
    real_index = idx_outer * inner_num + idx_inner;
    // get softmax result
    for (int j = 0; j < axis_size; ++j) {
      dout[real_index] *= sum_inv;
      real_index += inner_num;
    }
  }
  LITE_PARALLEL_END()
}

template <typename Dtype>
void softmax_axis4_sve(const Dtype* din,
                       Dtype* dout,
                       const int axis_size,
                       const int inner_num,
                       const int outer_num) {
  int compute_size = inner_num * outer_num;
  auto vone = svdup_n(static_cast<Dtype>(1));
  const auto all_true_pg = svptrue<Dtype>();
  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, compute_size, 0, svcnt<Dtype>()) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int real_index = idx_outer * inner_num + idx_inner;
    svbool_t pg = svwhilelt<Dtype>(i, compute_size);
    const Dtype* din_ptr0 = din + real_index;
    const Dtype* din_ptr1 = din_ptr0 + inner_num;
    const Dtype* din_ptr2 = din_ptr1 + inner_num;
    const Dtype* din_ptr3 = din_ptr2 + inner_num;
    auto vdata0 = svld1(pg, din_ptr0);
    auto vdata1 = svld1(pg, din_ptr1);
    auto vdata2 = svld1(pg, din_ptr2);
    auto vdata3 = svld1(pg, din_ptr3);
    Dtype* dout_ptr0 = dout + real_index;
    Dtype* dout_ptr1 = dout_ptr0 + inner_num;
    // get max
    auto vmax0 = svmax_m(pg, vdata0, vdata1);
    auto vmax1 = svmax_m(pg, vdata2, vdata3);
    Dtype* dout_ptr2 = dout_ptr1 + inner_num;
    Dtype* dout_ptr3 = dout_ptr2 + inner_num;
    auto vmax = svmax_m(pg, vmax0, vmax1);
    // sub, exp and sum
    auto vsum0 = svexp_z(pg, svsub_z(pg, vdata0, vmax));
    auto vsum1 = svexp_z(pg, svsub_z(pg, vdata1, vmax));
    auto vsum2 = svexp_z(pg, svsub_z(pg, vdata2, vmax));
    auto vsum3 = svexp_z(pg, svsub_z(pg, vdata3, vmax));

    auto vsum_0 = svadd_m(pg, vsum0, vsum1);
    auto vsum_1 = svadd_m(pg, vsum2, vsum3);
    auto vsum = svadd_m(pg, vsum_0, vsum_1);
    auto vinf = svdiv_z(pg, vone, vsum);
    auto vout0 = svmul_z(pg, vsum0, vinf);
    auto vout1 = svmul_z(pg, vsum1, vinf);
    auto vout2 = svmul_z(pg, vsum2, vinf);
    auto vout3 = svmul_z(pg, vsum3, vinf);
    svst1(pg, dout_ptr0, vout0);
    svst1(pg, dout_ptr1, vout1);
    svst1(pg, dout_ptr2, vout2);
    svst1(pg, dout_ptr3, vout3);
  }
  LITE_PARALLEL_END()
}

template <typename Dtype>
void softmax_inner1_sve(const Dtype* din,
                        Dtype* dout,
                        const int outer_size,
                        const int axis_size) {
  int out_cnt = (outer_size >> 2) << 2;
  auto vone = svdup_n(static_cast<Dtype>(1));
  const auto all_true_pg = svptrue<Dtype>();
  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, outer_size - 3, 0, 4) {
    auto index = i * axis_size;
    const Dtype* din_ptr0 = din + index;
    const Dtype* din_ptr1 = din_ptr0 + axis_size;
    const Dtype* din_ptr2 = din_ptr1 + axis_size;
    const Dtype* din_ptr3 = din_ptr2 + axis_size;
    const Dtype* din_max_ptr0 = din_ptr0;
    const Dtype* din_max_ptr1 = din_ptr1;
    const Dtype* din_max_ptr2 = din_ptr2;
    const Dtype* din_max_ptr3 = din_ptr3;
    int x = 0;
    auto pg0 = svwhilelt<Dtype>(x, axis_size);
    auto vec_max0 = svdup_n(static_cast<Dtype>(-FLT_MAX));
    auto vec_max1 = svdup_n(static_cast<Dtype>(-FLT_MAX));
    auto vec_max2 = svdup_n(static_cast<Dtype>(-FLT_MAX));
    auto vec_max3 = svdup_n(static_cast<Dtype>(-FLT_MAX));
    for (int j = 0; j < axis_size; j += svcnt<Dtype>()) {
      pg0 = svwhilelt<Dtype>(j, axis_size);
      auto vdata0 = svld1(pg0, din_max_ptr0);
      auto vdata1 = svld1(pg0, din_max_ptr1);
      auto vdata2 = svld1(pg0, din_max_ptr2);
      auto vdata3 = svld1(pg0, din_max_ptr3);
      // get max
      vec_max0 = svmax_m(pg0, vec_max0, vdata0);
      vec_max1 = svmax_m(pg0, vec_max1, vdata1);
      vec_max2 = svmax_m(pg0, vec_max2, vdata2);
      vec_max3 = svmax_m(pg0, vec_max3, vdata3);
      din_max_ptr0 += svcnt<Dtype>();
      din_max_ptr1 += svcnt<Dtype>();
      din_max_ptr2 += svcnt<Dtype>();
      din_max_ptr3 += svcnt<Dtype>();
    }
    Dtype vmax_0 = svmaxv(all_true_pg, vec_max0);
    Dtype vmax_1 = svmaxv(all_true_pg, vec_max1);
    Dtype vmax_2 = svmaxv(all_true_pg, vec_max2);
    Dtype vmax_3 = svmaxv(all_true_pg, vec_max3);
    // sub, exp and sum
    x = 0;
    din_max_ptr0 = din_ptr0;
    din_max_ptr1 = din_ptr1;
    din_max_ptr2 = din_ptr2;
    din_max_ptr3 = din_ptr3;
    Dtype* dout_ptr0 = dout + index;
    Dtype* dout_ptr1 = dout_ptr0 + axis_size;
    Dtype* dout_ptr2 = dout_ptr1 + axis_size;
    Dtype* dout_ptr3 = dout_ptr2 + axis_size;
    auto vsum0 = svdup_n(static_cast<Dtype>(0));
    auto vsum1 = svdup_n(static_cast<Dtype>(0));
    auto vsum2 = svdup_n(static_cast<Dtype>(0));
    auto vsum3 = svdup_n(static_cast<Dtype>(0));
    auto vmax0 = svdup_n(vmax_0);
    auto vmax1 = svdup_n(vmax_1);
    auto vmax2 = svdup_n(vmax_2);
    auto vmax3 = svdup_n(vmax_3);
    for (int j = 0; j < axis_size; j += svcnt<Dtype>()) {
      auto pg0 = svwhilelt<Dtype>(j, axis_size);
      auto vsub_exp0 =
          svexp_z(pg0, svsub_z(pg0, svld1(pg0, din_max_ptr0), vmax0));
      auto vsub_exp1 =
          svexp_z(pg0, svsub_z(pg0, svld1(pg0, din_max_ptr1), vmax1));
      auto vsub_exp2 =
          svexp_z(pg0, svsub_z(pg0, svld1(pg0, din_max_ptr2), vmax2));
      auto vsub_exp3 =
          svexp_z(pg0, svsub_z(pg0, svld1(pg0, din_max_ptr3), vmax3));
      vsum0 = svadd_m(pg0, vsum0, vsub_exp0);
      vsum1 = svadd_m(pg0, vsum1, vsub_exp1);
      vsum2 = svadd_m(pg0, vsum2, vsub_exp2);
      vsum3 = svadd_m(pg0, vsum3, vsub_exp3);
      din_max_ptr0 += svcnt<Dtype>();
      din_max_ptr1 += svcnt<Dtype>();
      din_max_ptr2 += svcnt<Dtype>();
      din_max_ptr3 += svcnt<Dtype>();
      svst1(pg0, dout_ptr0, vsub_exp0);
      svst1(pg0, dout_ptr1, vsub_exp1);
      svst1(pg0, dout_ptr2, vsub_exp2);
      svst1(pg0, dout_ptr3, vsub_exp3);
      dout_ptr0 += svcnt<Dtype>();
      dout_ptr1 += svcnt<Dtype>();
      dout_ptr2 += svcnt<Dtype>();
      dout_ptr3 += svcnt<Dtype>();
    }
    auto vsum_0 = svaddv(all_true_pg, vsum0);
    auto vsum_1 = svaddv(all_true_pg, vsum1);
    auto vsum_2 = svaddv(all_true_pg, vsum2);
    auto vsum_3 = svaddv(all_true_pg, vsum3);
    auto vinf0 = svdiv_z(all_true_pg, vone, svdup_n(vsum_0));
    auto vinf1 = svdiv_z(all_true_pg, vone, svdup_n(vsum_1));
    auto vinf2 = svdiv_z(all_true_pg, vone, svdup_n(vsum_2));
    auto vinf3 = svdiv_z(all_true_pg, vone, svdup_n(vsum_3));
    dout_ptr0 = dout + index;
    dout_ptr1 = dout_ptr0 + axis_size;
    dout_ptr2 = dout_ptr1 + axis_size;
    dout_ptr3 = dout_ptr2 + axis_size;
    for (int j = 0; j < axis_size; j += svcnt<Dtype>()) {
      auto pg0 = svwhilelt<Dtype>(j, axis_size);
      auto vsub_exp0 = svmul_z(pg0, svld1(pg0, dout_ptr0), vinf0);
      auto vsub_exp1 = svmul_z(pg0, svld1(pg0, dout_ptr1), vinf1);
      auto vsub_exp2 = svmul_z(pg0, svld1(pg0, dout_ptr2), vinf2);
      auto vsub_exp3 = svmul_z(pg0, svld1(pg0, dout_ptr3), vinf3);
      svst1(pg0, dout_ptr0, vsub_exp0);
      svst1(pg0, dout_ptr1, vsub_exp1);
      svst1(pg0, dout_ptr2, vsub_exp2);
      svst1(pg0, dout_ptr3, vsub_exp3);
      dout_ptr0 += svcnt<Dtype>();
      dout_ptr1 += svcnt<Dtype>();
      dout_ptr2 += svcnt<Dtype>();
      dout_ptr3 += svcnt<Dtype>();
    }
  }
  LITE_PARALLEL_COMMON_END()
  LITE_PARALLEL_COMMON_BEGIN(i, tid, outer_size, out_cnt, 1) {
    auto index = i * axis_size;
    const Dtype* din_ptr0 = din + index;
    const Dtype* din_max_ptr0 = din_ptr0;
    int x = 0;
    auto pg0 = svwhilelt<Dtype>(x, axis_size);
    auto vec_max0 = svdup_n(static_cast<Dtype>(-FLT_MAX));
    for (int j = 0; j < axis_size; j += svcnt<Dtype>()) {
      pg0 = svwhilelt<Dtype>(j, axis_size);
      auto vdata0 = svld1(pg0, din_max_ptr0);
      // get max
      vec_max0 = svmax_m(pg0, vec_max0, vdata0);
      din_max_ptr0 += svcnt<Dtype>();
    }
    Dtype vmax_0 = svmaxv(all_true_pg, vec_max0);
    // sub, exp and sum
    x = 0;
    din_max_ptr0 = din_ptr0;
    Dtype* dout_ptr0 = dout + index;
    auto vsum0 = svdup_n(static_cast<Dtype>(0));
    auto vmax0 = svdup_n(vmax_0);
    for (int j = 0; j < axis_size; j += svcnt<Dtype>()) {
      auto pg0 = svwhilelt<Dtype>(j, axis_size);
      auto vsub_exp0 =
          svexp_z(pg0, svsub_z(pg0, svld1(pg0, din_max_ptr0), vmax0));

      vsum0 = svadd_m(pg0, vsum0, vsub_exp0);
      din_max_ptr0 += svcnt<Dtype>();
      svst1(pg0, dout_ptr0, vsub_exp0);
      dout_ptr0 += svcnt<Dtype>();
    }
    auto vsum_0 = svaddv(all_true_pg, vsum0);
    auto vinf0 = svdiv_z(all_true_pg, vone, svdup_n(vsum_0));
    dout_ptr0 = dout + index;
    for (int j = 0; j < axis_size; j += svcnt<Dtype>()) {
      auto pg0 = svwhilelt<Dtype>(j, axis_size);
      auto vsub_exp0 = svmul_z(pg0, svld1(pg0, dout_ptr0), vinf0);
      svst1(pg0, dout_ptr0, vsub_exp0);
      dout_ptr0 += svcnt<Dtype>();
    }
  }
  LITE_PARALLEL_COMMON_END()
}

template void softmax_basic_sve<float>(const float* din,
                                       float* dout,
                                       const int axis_size,
                                       const int inner_num,
                                       const int outer_num);

template void softmax_axis4_sve<float>(const float* din,
                                       float* dout,
                                       const int axis_size,
                                       const int inner_num,
                                       const int outer_num);

template void softmax_inner1_sve<float>(const float* din,
                                        float* dout,
                                        const int outer_size,
                                        const int axis_size);

#ifdef ENABLE_ARM_FP16
template void softmax_basic_sve<float16_t>(const float16_t* din,
                                           float16_t* dout,
                                           const int axis_size,
                                           const int inner_num,
                                           const int outer_num);

template void softmax_axis4_sve<float16_t>(const float16_t* din,
                                           float16_t* dout,
                                           const int axis_size,
                                           const int inner_num,
                                           const int outer_num);

template void softmax_inner1_sve<float16_t>(const float16_t* din,
                                            float16_t* dout,
                                            const int outer_size,
                                            const int axis_size);
#endif

}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
