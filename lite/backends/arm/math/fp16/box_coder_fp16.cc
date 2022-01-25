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

#include "lite/backends/arm/math/fp16/box_coder_fp16.h"
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void encode_bbox_center_kernel(const int batch_num,  // N
                               const float16_t* target_box_data,
                               const float16_t* prior_box_data,
                               const float16_t* variance,
                               const bool var_len4,
                               const bool normalized,
                               const int num_priors,  // M
                               float16_t* output) {
  int len = 4;
  int size = batch_num * num_priors;
  float16_t norm_value = (normalized == false) ? 1 : 0;
  if (var_len4) {
    LITE_PARALLEL_BEGIN(k, tid, size) {
      int i = k / num_priors;
      int j = k % num_priors;
      size_t index_i = i * len;
      size_t index_j = j * len;
      size_t offset = k * len;
      float16_t prior_box_width =
          prior_box_data[index_j + 2] - prior_box_data[index_j] + norm_value;
      float16_t prior_box_height = prior_box_data[index_j + 3] -
                                   prior_box_data[index_j + 1] + norm_value;
      float16_t prior_box_center_x =
          prior_box_data[index_j] + prior_box_width / 2;
      float16_t prior_box_center_y =
          prior_box_data[index_j + 1] + prior_box_height / 2;

      float16_t target_box_center_x =
          (target_box_data[index_i + 2] + target_box_data[index_i]) / 2;
      float16_t target_box_center_y =
          (target_box_data[index_i + 3] + target_box_data[index_i + 1]) / 2;
      float16_t target_box_width =
          target_box_data[index_i + 2] - target_box_data[index_i] + norm_value;
      float16_t target_box_height = target_box_data[index_i + 3] -
                                    target_box_data[index_i + 1] + norm_value;

      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] = static_cast<float16_t>(
          std::log(std::fabs(target_box_width / prior_box_width)));
      output[offset + 3] = static_cast<float16_t>(
          std::log(std::fabs(target_box_height / prior_box_height)));
      output[offset] /= variance[0];
      output[offset + 1] /= variance[1];
      output[offset + 2] /= variance[2];
      output[offset + 3] /= variance[3];
    }
    LITE_PARALLEL_END()
  } else {
    LITE_PARALLEL_BEGIN(k, tid, size) {
      int i = k / num_priors;
      int j = k % num_priors;
      size_t index_i = i * len;
      size_t index_j = j * len;
      size_t offset = k * len;
      float16_t prior_box_width =
          prior_box_data[index_j + 2] - prior_box_data[index_j] + norm_value;
      float16_t prior_box_height = prior_box_data[index_j + 3] -
                                   prior_box_data[index_j + 1] + norm_value;
      float16_t prior_box_center_x =
          prior_box_data[index_j] + prior_box_width / 2;
      float16_t prior_box_center_y =
          prior_box_data[index_j + 1] + prior_box_height / 2;

      float16_t target_box_center_x =
          (target_box_data[index_i + 2] + target_box_data[index_i]) / 2;
      float16_t target_box_center_y =
          (target_box_data[index_i + 3] + target_box_data[index_i + 1]) / 2;
      float16_t target_box_width =
          target_box_data[index_i + 2] - target_box_data[index_i] + norm_value;
      float16_t target_box_height = target_box_data[index_i + 3] -
                                    target_box_data[index_i + 1] + norm_value;

      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] = static_cast<float16_t>(
          std::log(std::fabs(target_box_width / prior_box_width)));
      output[offset + 3] = static_cast<float16_t>(
          std::log(std::fabs(target_box_height / prior_box_height)));
      output[offset] /= variance[index_j];
      output[offset + 1] /= variance[index_j + 1];
      output[offset + 2] /= variance[index_j + 2];
      output[offset + 3] /= variance[index_j + 3];
    }
    LITE_PARALLEL_END()
  }
}

void decode_bbox_center_kernel(const int batch_num,
                               const float16_t* loc_data,
                               const float16_t* prior_data,
                               const float16_t* variance,
                               const bool var_len4,
                               const int num_priors,
                               const bool normalized,
                               float16_t* bbox_data) {
  int cnt = num_priors / 8;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vvar
  float16_t norm_value = (normalized == false) ? 1.f : 0.f;
  float16x8_t vhalf = vdupq_n_f16(0.5f);
  float16x8_t vnormalized = vdupq_n_f16(norm_value);
  int len_batch = num_priors * 4;
  for (int n = 0; n < batch_num; ++n) {
    const float16_t* ptr_loc_batch = loc_data + n * len_batch;
    float16_t* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i * 32;
      int var_idx = idx, prior_idx = idx;
      if (var_len4) {
        var_idx = 0;
      }

      const float16_t* ptr_loc = ptr_loc_batch + idx;
      const float16_t* ptr_prior = prior_data + prior_idx;
      const float16_t* ptr_var = variance + var_idx;

      float16_t* ptr_bbox = ptr_bbox_batch + idx;

      float16x8x4_t vprior = vld4q_f16(ptr_prior);
      float16x8x4_t vloc = vld4q_f16(ptr_loc);
      float16x8x4_t vvar;
      if (var_len4) {
        float16x8_t v0 = vdupq_n_f16(ptr_var[0]);
        float16x8_t v1 = vdupq_n_f16(ptr_var[1]);
        float16x8_t v2 = vdupq_n_f16(ptr_var[2]);
        float16x8_t v3 = vdupq_n_f16(ptr_var[3]);
        vvar = {v0, v1, v2, v3};
      } else {
        vvar = vld4q_f16(ptr_var);
      }
      float16x8_t vprior_width1 = vsubq_f16(vprior.val[2], vprior.val[0]);
      float16x8_t vprior_height1 = vsubq_f16(vprior.val[3], vprior.val[1]);
      float16x8_t vprior_width = vaddq_f16(vprior_width1, vnormalized);
      float16x8_t vprior_height = vaddq_f16(vprior_height1, vnormalized);

      float16x8_t vprior_cx =
          vaddq_f16(vprior.val[0], vmulq_f16(vprior_width, vhalf));
      float16x8_t vprior_cy =
          vaddq_f16(vprior.val[1], vmulq_f16(vprior_height, vhalf));

      vloc.val[0] = vmulq_f16(vloc.val[0], vvar.val[0]);
      vloc.val[1] = vmulq_f16(vloc.val[1], vvar.val[1]);
      vloc.val[2] = vmulq_f16(vloc.val[2], vvar.val[2]);
      vloc.val[3] = vmulq_f16(vloc.val[3], vvar.val[3]);

      float16x8_t vdec_bbx_cx =
          vaddq_f16(vmulq_f16(vloc.val[0], vprior_width), vprior_cx);
      float16x8_t vdec_bbx_cy =
          vaddq_f16(vmulq_f16(vloc.val[1], vprior_height), vprior_cy);
      float16x8_t vdec_bbx_w = expq_ps_f16(vloc.val[2]);
      float16x8_t vdec_bbx_h = expq_ps_f16(vloc.val[3]);
      vprior_width = vmulq_f16(vprior_width, vhalf);
      vprior_height = vmulq_f16(vprior_height, vhalf);
      vdec_bbx_w = vmulq_f16(vdec_bbx_w, vprior_width);
      vdec_bbx_h = vmulq_f16(vdec_bbx_h, vprior_height);

      vloc.val[0] = vsubq_f16(vdec_bbx_cx, vdec_bbx_w);
      vloc.val[1] = vsubq_f16(vdec_bbx_cy, vdec_bbx_h);
      vloc.val[2] = vaddq_f16(vdec_bbx_cx, vsubq_f16(vdec_bbx_w, vnormalized));
      vloc.val[3] = vaddq_f16(vdec_bbx_cy, vsubq_f16(vdec_bbx_h, vnormalized));

      vst4q_f16(ptr_bbox, vloc);
    }
    LITE_PARALLEL_END()

    LITE_PARALLEL_COMMON_BEGIN(i, tid, num_priors, cnt * 8, 1) {
      int idx = i * 4;
      int var_idx = idx, prior_idx = idx;
      if (var_len4) {
        var_idx = 0;
      }

      float16_t p_xmin = prior_data[prior_idx];
      float16_t p_ymin = prior_data[prior_idx + 1];
      float16_t p_xmax = prior_data[prior_idx + 2];
      float16_t p_ymax = prior_data[prior_idx + 3];
      float16_t prior_width = p_xmax - p_xmin + norm_value;
      float16_t prior_height = p_ymax - p_ymin + norm_value;
      float16_t prior_center_x = p_xmin + prior_width / 2.f;
      float16_t prior_center_y = p_ymin + prior_height / 2.f;

      float16_t xmin = ptr_loc_batch[idx];
      float16_t ymin = ptr_loc_batch[idx + 1];
      float16_t xmax = ptr_loc_batch[idx + 2];
      float16_t ymax = ptr_loc_batch[idx + 3];

      //! variance is encoded in target, we simply need to retore the offset
      //! predictions.
      float16_t decode_bbox_center_x =
          variance[var_idx] * xmin * prior_width + prior_center_x;
      float16_t decode_bbox_center_y =
          variance[var_idx + 1] * ymin * prior_height + prior_center_y;
      float16_t decode_bbox_width = static_cast<float16_t>(
          expf(variance[var_idx + 2] * xmax) * prior_width);
      float16_t decode_bbox_height = static_cast<float16_t>(
          expf(variance[var_idx + 3] * ymax) * prior_height);

      ptr_bbox_batch[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
      ptr_bbox_batch[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
      ptr_bbox_batch[idx + 2] =
          decode_bbox_center_x + decode_bbox_width / 2.f - norm_value;
      ptr_bbox_batch[idx + 3] =
          decode_bbox_center_y + decode_bbox_height / 2.f - norm_value;
    }
    LITE_PARALLEL_END()
  }
}

void decode_center_size_axis_1(const int var_size,
                               const int row,
                               const int col,
                               const int len,
                               const float16_t* target_box_data,
                               const float16_t* prior_box_data,
                               const float16_t* prior_box_var_data,
                               const bool normalized,
                               const std::vector<float16_t> variance,
                               float16_t* output) {
  float16_t norm_value = (normalized == false) ? 1.f : 0.f;
  int size = row * col;
  LITE_PARALLEL_BEGIN(k, tid, size) {
    int i = k / col;
    int j = k % col;
    float16_t var_data[4] = {1., 1., 1., 1.};
    float16_t* var_ptr = var_data;
    size_t offset = i * col * len + j * len;
    int prior_box_offset = i * len;  // axis == 0 ? j * len : i * len;

    float16_t prior_box_width = prior_box_data[prior_box_offset + 2] -
                                prior_box_data[prior_box_offset] + norm_value;
    float16_t prior_box_height = prior_box_data[prior_box_offset + 3] -
                                 prior_box_data[prior_box_offset + 1] +
                                 norm_value;
    float16_t prior_box_center_x =
        prior_box_data[prior_box_offset] + prior_box_width / 2;
    float16_t prior_box_center_y =
        prior_box_data[prior_box_offset + 1] + prior_box_height / 2;

    float16_t target_box_center_x = 0, target_box_center_y = 0;
    float16_t target_box_width = 0, target_box_height = 0;
    int prior_var_offset = i * len;  // axis == 0 ? j * len : i * len;
    if (var_size == 2) {
      std::memcpy(var_ptr,
                  prior_box_var_data + prior_var_offset,
                  4 * sizeof(float16_t));
    } else if (var_size == 1) {
      var_ptr = const_cast<float16_t*>(variance.data());
    }
    float box_var_x = *var_ptr;
    float box_var_y = *(var_ptr + 1);
    float box_var_w = *(var_ptr + 2);
    float box_var_h = *(var_ptr + 3);

    target_box_center_x =
        box_var_x * target_box_data[offset] * prior_box_width +
        prior_box_center_x;
    target_box_center_y =
        box_var_y * target_box_data[offset + 1] * prior_box_height +
        prior_box_center_y;
    target_box_width = static_cast<float16_t>(
        std::exp(box_var_w * target_box_data[offset + 2]) * prior_box_width);
    target_box_height = static_cast<float16_t>(
        std::exp(box_var_h * target_box_data[offset + 3]) * prior_box_height);

    output[offset] = target_box_center_x - target_box_width / 2;
    output[offset + 1] = target_box_center_y - target_box_height / 2;
    output[offset + 2] =
        target_box_center_x + target_box_width / 2 - norm_value;
    output[offset + 3] =
        target_box_center_y + target_box_height / 2 - norm_value;
  }
  LITE_PARALLEL_END()
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
