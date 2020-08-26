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

#include "lite/backends/arm/math/box_coder.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void decode_bbox_center_variance_kernel(const int batch_num,
                                        const float* loc_data,
                                        const float* prior_data,
                                        const float* variance,
                                        const int num_priors,
                                        float* bbox_data) {
  int cnt = num_priors / 4;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vvar
  float32x4_t vhalf = vdupq_n_f32(0.5f);

  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
    for (int i = 0; i < cnt; ++i) {
      int idx = i * 16;
      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4x4_t vprior = vld4q_f32(ptr_prior);
      float32x4x4_t vloc = vld4q_f32(ptr_loc);
      float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
      float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);
      float32x4_t vprior_cx =
          vmulq_f32(vaddq_f32(vprior.val[0], vprior.val[2]), vhalf);
      float32x4_t vprior_cy =
          vmulq_f32(vaddq_f32(vprior.val[1], vprior.val[3]), vhalf);

      float32x4_t vdec_bbx_cx =
          vaddq_f32(vmulq_f32(vloc.val[0], vprior_width), vprior_cx);
      float32x4_t vdec_bbx_cy =
          vaddq_f32(vmulq_f32(vloc.val[1], vprior_height), vprior_cy);
      float32x4_t vdec_bbx_w = exp_ps(vloc.val[2]);
      float32x4_t vdec_bbx_h = exp_ps(vloc.val[3]);
      vprior_width = vmulq_f32(vprior_width, vhalf);
      vprior_height = vmulq_f32(vprior_height, vhalf);
      vdec_bbx_w = vmulq_f32(vdec_bbx_w, vprior_width);
      vdec_bbx_h = vmulq_f32(vdec_bbx_h, vprior_height);

      vloc.val[0] = vsubq_f32(vdec_bbx_cx, vdec_bbx_w);
      vloc.val[1] = vsubq_f32(vdec_bbx_cy, vdec_bbx_h);
      vloc.val[2] = vaddq_f32(vdec_bbx_cx, vdec_bbx_w);
      vloc.val[3] = vaddq_f32(vdec_bbx_cy, vdec_bbx_h);

      vst4q_f32(ptr_bbox, vloc);
    }
#pragma omp parallel for
    for (int i = cnt * 4; i < num_priors; i++) {
      int idx = i * 4;
      float p_xmin = prior_data[idx];
      float p_ymin = prior_data[idx + 1];
      float p_xmax = prior_data[idx + 2];
      float p_ymax = prior_data[idx + 3];
      float prior_width = p_xmax - p_xmin;
      float prior_height = p_ymax - p_ymin;
      float prior_center_x = (p_xmin + p_xmax) / 2.f;
      float prior_center_y = (p_ymin + p_ymax) / 2.f;

      float xmin = ptr_loc_batch[idx];
      float ymin = ptr_loc_batch[idx + 1];
      float xmax = ptr_loc_batch[idx + 2];
      float ymax = ptr_loc_batch[idx + 3];

      //! variance is encoded in target, we simply need to retore the offset
      //! predictions.
      float decode_bbox_center_x = xmin * prior_width + prior_center_x;
      float decode_bbox_center_y = ymin * prior_height + prior_center_y;
      float decode_bbox_width = expf(xmax) * prior_width;
      float decode_bbox_height = expf(ymax) * prior_height;

      ptr_bbox_batch[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
      ptr_bbox_batch[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
      ptr_bbox_batch[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
      ptr_bbox_batch[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    }
  }
}

void decode_bbox_center_no_variance_kernel(const int batch_num,
                                           const float* loc_data,
                                           const float* prior_data,
                                           const float* variance,
                                           const int num_priors,
                                           const bool normalized,
                                           float* bbox_data) {
  int cnt = num_priors / 4;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vvar
  float32x4_t vhalf = vdupq_n_f32(0.5f);
  float norm_value = (normalized == false);
  float32x4_t vnormalized = vdupq_n_f32(norm_value);
  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

#pragma omp parallel for
    for (int i = 0; i < cnt; ++i) {
      int idx = i * 16;

      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      const float* ptr_var = variance + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4x4_t vprior = vld4q_f32(ptr_prior);
      float32x4x4_t vloc = vld4q_f32(ptr_loc);
      float32x4x4_t vvar = vld4q_f32(ptr_var);
      float32x4_t vprior_width1 = vsubq_f32(vprior.val[2], vprior.val[0]);
      float32x4_t vprior_height1 = vsubq_f32(vprior.val[3], vprior.val[1]);
      float32x4_t vprior_width = vaddq_f32(vprior_width1, vnormalized);
      float32x4_t vprior_height = vaddq_f32(vprior_height1, vnormalized);

      float32x4_t vprior_cx =
          vaddq_f32(vprior.val[0], vmulq_f32(vprior_width, vhalf));
      float32x4_t vprior_cy =
          vaddq_f32(vprior.val[1], vmulq_f32(vprior_height, vhalf));

      vloc.val[0] = vmulq_f32(vloc.val[0], vvar.val[0]);
      vloc.val[1] = vmulq_f32(vloc.val[1], vvar.val[1]);
      vloc.val[2] = vmulq_f32(vloc.val[2], vvar.val[2]);
      vloc.val[3] = vmulq_f32(vloc.val[3], vvar.val[3]);

      float32x4_t vdec_bbx_cx =
          vaddq_f32(vmulq_f32(vloc.val[0], vprior_width), vprior_cx);
      float32x4_t vdec_bbx_cy =
          vaddq_f32(vmulq_f32(vloc.val[1], vprior_height), vprior_cy);
      float32x4_t vdec_bbx_w = exp_ps(vloc.val[2]);
      float32x4_t vdec_bbx_h = exp_ps(vloc.val[3]);
      vprior_width = vmulq_f32(vprior_width, vhalf);
      vprior_height = vmulq_f32(vprior_height, vhalf);
      vdec_bbx_w = vmulq_f32(vdec_bbx_w, vprior_width);
      vdec_bbx_h = vmulq_f32(vdec_bbx_h, vprior_height);

      vloc.val[0] = vsubq_f32(vdec_bbx_cx, vdec_bbx_w);
      vloc.val[1] = vsubq_f32(vdec_bbx_cy, vdec_bbx_h);
      vloc.val[2] = vaddq_f32(vdec_bbx_cx, vsubq_f32(vdec_bbx_w, vnormalized));
      vloc.val[3] = vaddq_f32(vdec_bbx_cy, vsubq_f32(vdec_bbx_h, vnormalized));

      vst4q_f32(ptr_bbox, vloc);
    }

#pragma omp parallel for
    for (int i = cnt * 4; i < num_priors; i++) {
      int idx = i * 4;
      float p_xmin = prior_data[idx];
      float p_ymin = prior_data[idx + 1];
      float p_xmax = prior_data[idx + 2];
      float p_ymax = prior_data[idx + 3];
      float prior_width = p_xmax - p_xmin + norm_value;
      float prior_height = p_ymax - p_ymin + norm_value;
      float prior_center_x = p_xmin + prior_width / 2.f;
      float prior_center_y = p_ymin + prior_height / 2.f;

      float xmin = ptr_loc_batch[idx];
      float ymin = ptr_loc_batch[idx + 1];
      float xmax = ptr_loc_batch[idx + 2];
      float ymax = ptr_loc_batch[idx + 3];

      //! variance is encoded in target, we simply need to retore the offset
      //! predictions.
      float decode_bbox_center_x =
          variance[idx] * xmin * prior_width + prior_center_x;
      float decode_bbox_center_y =
          variance[idx + 1] * ymin * prior_height + prior_center_y;
      float decode_bbox_width = expf(variance[idx + 2] * xmax) * prior_width;
      float decode_bbox_height = expf(variance[idx + 3] * ymax) * prior_height;

      ptr_bbox_batch[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
      ptr_bbox_batch[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
      ptr_bbox_batch[idx + 2] =
          decode_bbox_center_x + decode_bbox_width / 2.f - norm_value;
      ptr_bbox_batch[idx + 3] =
          decode_bbox_center_y + decode_bbox_height / 2.f - norm_value;
    }
  }
}

void decode_bboxes(const int batch_num,
                   const float* loc_data,
                   const float* prior_data,
                   const float* variance_data,
                   const std::string code_type,
                   const bool normalized,
                   const int num_priors,
                   float* bbox_data) {
  if (code_type == "encode_center_size") {
    decode_bbox_center_variance_kernel(
        batch_num, loc_data, prior_data, variance_data, num_priors, bbox_data);

  } else if (code_type == "decode_center_size") {
    decode_bbox_center_no_variance_kernel(batch_num,
                                          loc_data,
                                          prior_data,
                                          variance_data,
                                          num_priors,
                                          normalized,
                                          bbox_data);
  } else {
    LOG(FATAL) << "box_coder don't support this code_type: " << code_type;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
