/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "lite/backends/loongarch/jit/more/intrinsic/crf_decoding.h"
#include <limits>
#include "lite/backends/loongarch/cpu_info.h"
#include "lite/backends/loongarch/jit/registry.h"

namespace paddle {
namespace lite {
namespace jit {
namespace more {
namespace intrinsic {

void CRFDecoding(const int seq_len,
                 const float* x,
                 const float* w,
                 float* alpha,
                 int* track,
                 int tag_num) {
  const int step_size = LASX_FLOAT_BLOCK;
  const int end = tag_num / step_size;
  const int rest = tag_num % step_size;
  /* Setup the alpha initial value.*/
  int i_offset = 0;
  int last_offset = rest - step_size;
  for (int i = 0; i <= end; ++i) {
    // weights, input and alpha values.
    __m256 w_content, x_content, alpha_content;
    // Load the relevant data into the variables from un-aligned address.
    w_content = lasx_loadu_f32(w + i_offset);
    x_content = lasx_loadu_f32(x + i_offset);
    alpha_content = lasx_add_f32(w_content, x_content);
    lasx_storeu_f32(alpha + i_offset, alpha_content);
    i_offset += step_size;
    if (i == end - 1) {
      if (rest > 0) {
        i_offset += last_offset;
      } else {
        break;
      }
    }
  }
  // Use the column-major strategy to get the location of maximum score.
  int seq_offset = 0;
  constexpr int state_trans_base_idx = 2;
  for (int k = 1; k < seq_len; ++k) {
    int j_offset = 0;
    for (int j = 0; j <= end; ++j) {
/* Initialize the variables of maximum score and location.*/
      __m256 max_score = lasx_set1_f32(-std::numeric_limits<float>::max());
      __m256i max_j = lasx_set1_i32(0);
      /* Calculate the offset of transition_weights.*/
      int trans_offset = state_trans_base_idx * tag_num + j_offset;
      for (int i = 0; i < tag_num; ++i) {
/* Initalize the content of alpha variable with related offset.*/
        __m256 alpha_content = lasx_broadcast_1f32(alpha + seq_offset + i);
        /* Obtain the content of weights from un-aligned address.*/
        __m256 w_content = lasx_loadu_f32(w + trans_offset);
        __m256 score_v = lasx_add_f32(alpha_content, w_content);
        __m256i mask =
            lasx_castf32_m256i(lasx_xvfcmp_slt_s(max_score, score_v));
/* According to the mask value, update the index of the max_score.*/
        max_j = lasx_or_m256i(lasx_andnot_m256i(mask, max_j),
                                lasx_and_m256i(mask, lasx_set1_i32(i)));
        /* Update the max_score value.*/
        max_score = lasx_max_f32(max_score, score_v);

        trans_offset += tag_num;
      }
/* Update the alpha and track values. */
      __m256 x_content = lasx_loadu_f32(x + seq_offset + tag_num + j_offset);
      max_score = lasx_add_f32(max_score, x_content);
      lasx_storeu_f32(alpha + seq_offset + tag_num + j_offset, max_score);
      lasx_storeu_m256i(
          reinterpret_cast<__m256i*>(track + seq_offset + tag_num + j_offset),
          max_j);

      /* Calculate the offset of next step*/
      j_offset += step_size;
      if (j == end - 1) {
        if (rest > 0) {
          j_offset += last_offset;
        } else {
          break;
        }
      }
    }
    seq_offset += tag_num;
  }
}

bool CRFDecodingKernel::CanBeUsed(const int& d) const {
  constexpr int block = LASX_FLOAT_BLOCK;
  return loongarch::MayIUse(loongarch::lasx) && d >= block;
}

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace lite
}  // namespace paddle

namespace intrinsic = paddle::lite::jit::more::intrinsic;

REGISTER_JITKERNEL_MORE(kCRFDecoding, intrinsic, intrinsic::CRFDecodingKernel);
