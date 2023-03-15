// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/viterbi_decode.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "lite/backends/arm/math/argmax.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#define VEC_PROCESS_FP32(op, sym)                                       \
  void vector_##op(const float *a, const float *b, float *c, int num) { \
    int i = 0;                                                          \
    for (; i + 3 < num; i += 4) {                                       \
      float32x4_t vec_a = vld1q_f32(a + i);                             \
      float32x4_t vec_b = vld1q_f32(b + i);                             \
      vst1q_f32(c + i, v##op##q_f32(vec_a, vec_b));                     \
    }                                                                   \
    for (; i < num; i++) {                                              \
      c[i] = a[i] sym b[i];                                             \
    }                                                                   \
  }                                                                     \
  void vector_##op(float a, const float *b, float *c, int num) {        \
    int i = 0;                                                          \
    float32x4_t vec_a = vdupq_n_f32(a);                                 \
    for (; i + 3 < num; i += 4) {                                       \
      float32x4_t vec_b = vld1q_f32(b + i);                             \
      vst1q_f32(c + i, v##op##q_f32(vec_a, vec_b));                     \
    }                                                                   \
    for (; i < num; i++) {                                              \
      c[i] = a sym b[i];                                                \
    }                                                                   \
  }

#define VEC_PROCESS_S32(op, sym)                                  \
  void vector_##op(const int *a, const int *b, int *c, int num) { \
    int i = 0;                                                    \
    for (; i + 3 < num; i += 4) {                                 \
      int32x4_t vec_a = vld1q_s32(a + i);                         \
      int32x4_t vec_b = vld1q_s32(b + i);                         \
      vst1q_s32(c + i, v##op##q_s32(vec_a, vec_b));               \
    }                                                             \
    for (; i < num; i++) {                                        \
      c[i] = a[i] sym b[i];                                       \
    }                                                             \
  }                                                               \
  void vector_##op(int a, const int *b, int *c, int num) {        \
    int i = 0;                                                    \
    int32x4_t vec_a = vdupq_n_s32(a);                             \
    for (; i + 3 < num; i += 4) {                                 \
      int32x4_t vec_b = vld1q_s32(b + i);                         \
      vst1q_s32(c + i, v##op##q_s32(vec_a, vec_b));               \
    }                                                             \
    for (; i < num; i++) {                                        \
      c[i] = a sym b[i];                                          \
    }                                                             \
  }

void vector_mla(int *out, const int *a, const int *b, int *c, int num) {
  int i = 0;
  for (; i + 3 < num; i += 4) {
    int32x4_t vec_a = vld1q_s32(a + i);
    int32x4_t vec_b = vld1q_s32(b + i);
    int32x4_t vec_c = vld1q_s32(c + i);
    vst1q_s32(out + i, vmlaq_s32(vec_a, vec_b, vec_c));
  }
  for (; i < num; i++) {
    out[i] = a[i] + b[i] * c[i];
  }
}

VEC_PROCESS_FP32(add, +)
VEC_PROCESS_FP32(sub, -)
VEC_PROCESS_FP32(mul, *)
VEC_PROCESS_S32(add, +)
VEC_PROCESS_S32(sub, -)
VEC_PROCESS_S32(mul, *)

void arange(int *data, int end, int scale) {
  for (int i = 0; i < end; ++i) {
    data[i] = i * scale;
  }
}

void arg_max(const float *in_data,
             int *out_idx_data,
             float *out_data,
             int pre,
             int n,
             int post) {
  int height = pre * post;
  int width = n;
  for (int i = 0; i < height; ++i) {
    int h = i / post;
    int w = i % post;
    int max_idx = -1;
    float max_value = std::numeric_limits<float>::lowest();
    for (int j = 0; j < width; ++j) {
      if (in_data[h * width * post + j * post + w] > max_value) {
        max_value = in_data[h * width * post + j * post + w];
        max_idx = j;
      }
    }
    out_data[i] = max_value;
    out_idx_data[i] = max_idx;
  }
}

void transpose_s32_s64(int *input, int64_t *output, int x, int y) {
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      output[j * x + i] = static_cast<int64_t>(input[i * y + j]);
    }
  }
}

void viterbi_decode(
    const lite::Tensor &input,       // [batch, seq_len, n_labels] float
    const lite::Tensor &transition,  // [n_labels, n_labels] float
    const lite::Tensor &length,      // [batch] int64
    bool include_bos_eos_tag,
    Tensor *scores,  // [batch] float
    Tensor *path) {  // [batch, seq_len] int64
  auto x_dims = input.dims();
  auto batch = x_dims[0];
  auto seq_len = x_dims[1];
  auto n_labels = x_dims[2];
  int64_t max_seq_len = 0;
  auto input_ptr = input.data<float>();
  auto length_ptr = length.data<int64_t>();
  max_seq_len = *std::max_element(length_ptr, length_ptr + length.numel());

  if (max_seq_len == 0) {
    path->Resize({batch, 1});
    auto path_ptr = path->mutable_data<int64_t>();
    scores->Resize({batch});
    auto scores_ptr = scores->mutable_data<float>();
    return;
  }
  path->Resize({batch, max_seq_len});
  auto path_ptr = path->mutable_data<int64_t>();
  scores->Resize({batch});
  auto scores_ptr = scores->mutable_data<float>();

  // temporary buffer
  lite::Tensor float_mask;
  float_mask.Resize({batch});
  auto float_mask_ptr = float_mask.mutable_data<float>();
  lite::Tensor alpha;
  alpha.Resize({batch, n_labels});
  auto alpha_ptr = alpha.mutable_data<float>();
  lite::Tensor alpha_trn_sum;
  alpha_trn_sum.Resize({batch, n_labels, n_labels});
  auto alpha_trn_sum_ptr = alpha_trn_sum.mutable_data<float>();
  lite::Tensor alpha_max;
  alpha_max.Resize({batch, n_labels});
  auto alpha_max_ptr = alpha_max.mutable_data<float>();
  lite::Tensor history;
  history.Resize({max_seq_len - 1, batch, n_labels});
  auto history_ptr = history.mutable_data<int>();  // argmax_id
  lite::Tensor alpha_nxt;
  alpha_nxt.Resize({batch, n_labels});
  auto alpha_nxt_ptr = alpha_nxt.mutable_data<float>();
  // treat length_int64 as left_length_int32 for neon(v7/v8)
  lite::Tensor left_length;
  left_length.Resize({batch});
  auto left_length_ptr = left_length.mutable_data<int>();
  lite::Tensor last_ids;
  last_ids.Resize({batch});
  auto last_ids_ptr = last_ids.mutable_data<int>();
  lite::Tensor last_ids_tmp;
  last_ids_tmp.Resize({batch});
  auto last_ids_tmp_ptr = last_ids_tmp.mutable_data<int>();
  lite::Tensor int_mask;
  int_mask.Resize({batch});
  auto int_mask_ptr = int_mask.mutable_data<int>();
  lite::Tensor batch_path;
  batch_path.Resize({max_seq_len, batch});
  auto batch_path_ptr = batch_path.mutable_data<int>();
  lite::Tensor batch_offset;
  batch_offset.Resize({batch});
  auto batch_offset_ptr = batch_offset.mutable_data<int>();
  lite::Tensor gather_idx;
  gather_idx.Resize({batch});
  auto gather_idx_ptr = gather_idx.mutable_data<int>();
  lite::Tensor zero_len_mask;
  zero_len_mask.Resize({batch});
  auto zero_len_mask_ptr = zero_len_mask.mutable_data<int>();

  auto trans_ptr = transition.data<float>();
  auto res_trans = trans_ptr;
  auto stop_trans = trans_ptr + (n_labels - 2) * n_labels;
  auto start_trans = stop_trans + n_labels;
  float32x4_t vec_1_f32 = vdupq_n_f32(1.f);
  float32x4_t vec_0_f32 = vdupq_n_f32(0.f);
  int32x4_t vec_0_s32 = vdupq_n_s32(0);

  if (include_bos_eos_tag) {
    // GetMask
    for (int i = 0; i < batch; i++) {
      float_mask_ptr[i] = length_ptr[i] == 1 ? 1.f : 0.f;
    }
    // alpha{batch, n_labels} = float_mask{batch} * stop_trans{n_labels} +
    // logit0{batch, n_labels} + start_trans{1, n_labels}
    for (int i = 0; i < batch; i++) {
      const float *logit0 = input_ptr + i * seq_len * n_labels;
      float *alpha_ptr_tmp = alpha_ptr + i * n_labels;
      float32x4_t vec_mask = vdupq_n_f32(float_mask_ptr[i]);
      int j = 0;
      for (; j + 3 < n_labels; j += 4) {
        float32x4_t vec_logit0 = vld1q_f32(logit0 + j);
        float32x4_t vec_start_trans = vld1q_f32(start_trans + j);
        float32x4_t vec_stop_trans = vld1q_f32(stop_trans + j);
        float32x4_t vec_alpha = vmlaq_f32(
            vaddq_f32(vec_logit0, vec_start_trans), vec_stop_trans, vec_mask);
        vst1q_f32(alpha_ptr_tmp + j, vec_alpha);
      }
      for (; j < n_labels; j++) {
        alpha_ptr_tmp[j] =
            logit0[j] + start_trans[j] + float_mask_ptr[i] * stop_trans[j];
      }
    }
  } else {
    for (int i = 0; i < batch; i++) {
      memcpy(alpha_ptr + i * n_labels,
             input_ptr + i * seq_len * n_labels,
             n_labels * sizeof(float));
    }
  }

  // length -= 1
  for (int i = 0; i < batch; i++) {
    left_length_ptr[i] = static_cast<int>(length_ptr[i] - 1);
  }
  for (int sl = 1; sl < max_seq_len; sl++) {
    // alpha_trn_sum{batch, n_labels, n_labels} =
    // alpha{batch, n_labels, 1} + transition{n_labels, n_labels}
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < n_labels; j++) {
        auto trans_ptr_tmp = trans_ptr + j * n_labels;
        int k = 0;
        float *alpha_trn_sum_ptr_out =
            alpha_trn_sum_ptr + i * n_labels * n_labels + j * n_labels;
        float *alpha_tmp = alpha_ptr + i * n_labels + j;
        float32x4_t vec_alpha = vdupq_n_f32(alpha_tmp[0]);
        for (; k + 3 < n_labels; k += 4) {
          float32x4_t vec_trans = vld1q_f32(trans_ptr_tmp + k);
          vst1q_f32(alpha_trn_sum_ptr_out + k, vaddq_f32(vec_alpha, vec_trans));
        }
        for (; k < n_labels; k++) {
          alpha_trn_sum_ptr_out[k] = trans_ptr_tmp[k] + alpha_tmp[0];
        }
      }
    }
    arg_max(alpha_trn_sum_ptr,
            history_ptr + (sl - 1) * batch * n_labels,
            alpha_max_ptr,
            batch,
            n_labels,
            n_labels);
    // alpha_nxt = alpha_max{batch, n_labels} + logit{batch, n_labels}
    for (int i = 0; i < batch; i++) {
      const float *logit = input_ptr + i * seq_len * n_labels + sl * n_labels;
      float *alpha_max_ptr_tmp = alpha_max_ptr + i * n_labels;
      vector_add(
          logit, alpha_max_ptr_tmp, alpha_nxt_ptr + i * n_labels, n_labels);
    }
    // GetMask > 0
    int i = 0;
    for (; i + 3 < batch; i += 4) {
      uint32x4_t tmp_mask =
          vcgtq_s32(vld1q_s32(left_length_ptr + i), vec_0_s32);
      vst1q_f32(float_mask_ptr + i, vbslq_f32(tmp_mask, vec_1_f32, vec_0_f32));
    }
    for (; i < batch; i++) {
      float_mask_ptr[i] = left_length_ptr[i] > 0 ? 1.f : 0.f;
    }
    // alpha_nxt = alpha_nxt * float_mask
    for (int i = 0; i < batch; i++) {
      vector_mul(float_mask_ptr[i],
                 alpha_nxt_ptr + i * n_labels,
                 alpha_nxt_ptr + i * n_labels,
                 n_labels);
    }
    // float_mask = 1 - float_mask
    vector_sub(1, float_mask_ptr, float_mask_ptr, batch);
    // alpha = alpha * float_mask + alpha_nxt
    for (int i = 0; i < batch; i++) {
      int j = 0;
      float32x4_t vec_mask = vdupq_n_f32(float_mask_ptr[i]);
      for (; j + 3 < n_labels; j += 4) {
        int offset = i * n_labels + j;
        float32x4_t vec_alpha = vld1q_f32(alpha_ptr + offset);
        float32x4_t vec_nxt = vld1q_f32(alpha_nxt_ptr + offset);
        vst1q_f32(alpha_ptr + offset, vmlaq_f32(vec_nxt, vec_alpha, vec_mask));
      }
      for (; j < n_labels; j++) {
        int offset = i * n_labels + j;
        alpha_ptr[offset] =
            alpha_ptr[offset] * float_mask_ptr[i] + alpha_nxt_ptr[offset];
      }
    }
    if (include_bos_eos_tag) {
      // GetMask == 1
      for (int i = 0; i < batch; i++) {
        float_mask_ptr[i] = length_ptr[i] == 1 ? 1.f : 0.f;
      }
      // alpha{batch, n_labels} = float_mask{batch} * stop_trans{n_labels} +
      // alpha{batch, n_labels}
      for (int i = 0; i < batch; i++) {
        float *alpha_ptr_tmp = alpha_ptr + i * n_labels;
        float32x4_t vec_mask = vdupq_n_f32(float_mask_ptr[i]);
        int j = 0;
        for (; j + 3 < n_labels; j += 4) {
          float32x4_t vec_al = vld1q_f32(alpha_ptr_tmp + j);
          float32x4_t vec_stop_trans = vld1q_f32(stop_trans + j);
          float32x4_t vec_alpha = vmlaq_f32(vec_al, vec_stop_trans, vec_mask);
          vst1q_f32(alpha_ptr_tmp + j, vec_alpha);
        }
        for (; j < n_labels; j++) {
          alpha_ptr_tmp[j] =
              alpha_ptr_tmp[j] + float_mask_ptr[i] * stop_trans[j];
        }
      }
    }
    // length -= 1
    for (int i = 0; i < batch; i++) {
      left_length_ptr[i] = left_length_ptr[i] - 1;
    }
  }

  arg_max(alpha_ptr, last_ids_ptr, scores_ptr, batch, n_labels, 1);
  for (int i = 0; i < batch; i++) {
    int_mask_ptr[i] = left_length_ptr[i] >= 0 ? 1 : 0;
  }
  // last_ids_update = last_ids * tag_mask
  int last_ids_index = 1;
  int actual_len =
      std::min(static_cast<int>(seq_len), static_cast<int>(max_seq_len));
  vector_mul(last_ids_ptr,
             int_mask_ptr,
             batch_path_ptr + (actual_len - last_ids_index) * batch,
             batch);
  arange(batch_offset_ptr, batch, n_labels);
  for (int sl = max_seq_len - 2; sl >= 0; sl--) {
    ++last_ids_index;
    vector_add(1, left_length_ptr, left_length_ptr, batch);
    vector_add(batch_offset_ptr, last_ids_ptr, gather_idx_ptr, batch);
    auto last_ids_update =
        batch_path_ptr + (actual_len - last_ids_index) * batch;
    auto his_ptr = history_ptr + sl * (batch * n_labels);
    // gather
    for (int64_t i = 0; i < batch; ++i) {
      last_ids_update[i] = his_ptr[gather_idx_ptr[i]];
    }
    // GetMask > 0
    for (int i = 0; i < batch; i++) {
      int_mask_ptr[i] = left_length_ptr[i] > 0 ? 1 : 0;
    }
    vector_mul(last_ids_update, int_mask_ptr, last_ids_update, batch);
    // GetMask == 0
    for (int i = 0; i < batch; i++) {
      zero_len_mask_ptr[i] = left_length_ptr[i] == 0 ? 1 : 0;
    }
    vector_mul(last_ids_ptr, zero_len_mask_ptr, last_ids_tmp_ptr, batch);
    vector_sub(1, zero_len_mask_ptr, zero_len_mask_ptr, batch);
    vector_mla(last_ids_update,
               last_ids_tmp_ptr,
               last_ids_update,
               zero_len_mask_ptr,
               batch);
    // GetMask < 0
    for (int i = 0; i < batch; i++) {
      int_mask_ptr[i] = left_length_ptr[i] < 0 ? 1 : 0;
    }
    vector_mla(
        last_ids_ptr, last_ids_update, last_ids_ptr, int_mask_ptr, batch);
  }
  // todo: vmovl optimize
  transpose_s32_s64(batch_path_ptr, path_ptr, max_seq_len, batch);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
