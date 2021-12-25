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

#include "lite/backends/arm/math/decode_bboxes.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void decode_bbox_corner_variance_kernel(const int batch_num,
                                        const T* loc_data,
                                        const T* prior_data,
                                        const T* variance,
                                        const int num_priors,
                                        const bool share_location,
                                        const int num_loc_classes,
                                        const int background_label_id,
                                        T* bbox_data);

template <typename T>
void decode_bbox_corner_no_variance_kernel(const int batch_num,
                                           const T* loc_data,
                                           const T* prior_data,
                                           const T* variance,
                                           const int num_priors,
                                           const bool share_location,
                                           const int num_loc_classes,
                                           const int background_label_id,
                                           T* bbox_data);

template <typename T>
void decode_bbox_center_variance_kernel(const int batch_num,
                                        const T* loc_data,
                                        const T* prior_data,
                                        const T* variance,
                                        const int num_priors,
                                        const bool share_location,
                                        const int num_loc_classes,
                                        const int background_label_id,
                                        T* bbox_data);

template <typename T>
void decode_bbox_center_no_variance_kernel(const int batch_num,
                                           const float* loc_data,
                                           const float* prior_data,
                                           const float* variance,
                                           const int num_priors,
                                           const bool share_location,
                                           const int num_loc_classes,
                                           const int background_label_id,
                                           float* bbox_data);

template <typename T>
void decode_bbox_corner_size_variance_kernel(const int batch_num,
                                             const T* loc_data,
                                             const T* prior_data,
                                             const T* variance,
                                             const int num_priors,
                                             const bool share_location,
                                             const int num_loc_classes,
                                             const int background_label_id,
                                             T* bbox_data);

template <typename T>
void decode_bbox_corner_size_no_variance_kernel(const int batch_num,
                                                const T* loc_data,
                                                const T* prior_data,
                                                const T* variance,
                                                const int num_priors,
                                                const bool share_location,
                                                const int num_loc_classes,
                                                const int background_label_id,
                                                T* bbox_data);

template <>
void decode_bbox_corner_variance_kernel<float>(const int batch_num,
                                               const float* loc_data,
                                               const float* prior_data,
                                               const float* variance,
                                               const int num_priors,
                                               const bool share_location,
                                               const int num_loc_classes,
                                               const int background_label_id,
                                               float* bbox_data) {
  if (!share_location) {
    CHECK_EQ(share_location, true)
        << "ERROR: decode boxes without share_location is unimplemented\n";
    return;
  }

  int cnt = num_priors / 4;
  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i * 16;
      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4_t vloc1 = vld1q_f32(ptr_loc);
      float32x4_t vloc2 = vld1q_f32(ptr_loc + 4);
      float32x4_t vloc3 = vld1q_f32(ptr_loc + 8);
      float32x4_t vloc4 = vld1q_f32(ptr_loc + 12);

      float32x4_t vprior1 = vld1q_f32(ptr_prior);
      float32x4_t vprior2 = vld1q_f32(ptr_prior + 4);
      float32x4_t vprior3 = vld1q_f32(ptr_prior + 8);
      float32x4_t vprior4 = vld1q_f32(ptr_prior + 12);

      vst1q_f32(ptr_bbox, vaddq_f32(vloc1, vprior1));
      vst1q_f32(ptr_bbox + 4, vaddq_f32(vloc2, vprior2));
      vst1q_f32(ptr_bbox + 8, vaddq_f32(vloc3, vprior3));
      vst1q_f32(ptr_bbox + 12, vaddq_f32(vloc4, vprior4));
    }
    LITE_PARALLEL_END()

    LITE_PARALLEL_COMMON_BEGIN(i, tid, num_priors, cnt * 4, 1) {
      int idx = i * 4;
      float32x4_t vloc = vld1q_f32(ptr_loc_batch + idx);
      float32x4_t vprior = vld1q_f32(prior_data + idx);
      vst1q_f32(ptr_bbox_batch + idx, vaddq_f32(vloc, vprior));
    }
    LITE_PARALLEL_END()
  }
}

template <>
void decode_bbox_corner_no_variance_kernel<float>(const int batch_num,
                                                  const float* loc_data,
                                                  const float* prior_data,
                                                  const float* variance,
                                                  const int num_priors,
                                                  const bool share_location,
                                                  const int num_loc_classes,
                                                  const int background_label_id,
                                                  float* bbox_data) {
  if (!share_location) {
    CHECK_EQ(share_location, true)
        << "ERROR: decode boxes without share_location is unimplemented\n";
    return;
  }

  int cnt = num_priors / 4;
  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i * 16;
      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      const float* ptr_var = variance + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4_t vloc1 = vld1q_f32(ptr_loc);
      float32x4_t vprior1 = vld1q_f32(ptr_prior);
      float32x4_t vvar1 = vld1q_f32(ptr_var);
      float32x4_t vout1 = vmulq_f32(vloc1, vvar1);

      float32x4_t vloc2 = vld1q_f32(ptr_loc + 4);
      float32x4_t vprior2 = vld1q_f32(ptr_prior + 4);
      float32x4_t vvar2 = vld1q_f32(ptr_var + 4);
      float32x4_t vout2 = vmulq_f32(vloc2, vvar2);

      float32x4_t vloc3 = vld1q_f32(ptr_loc + 8);
      float32x4_t vprior3 = vld1q_f32(ptr_prior + 8);
      float32x4_t vvar3 = vld1q_f32(ptr_var + 8);
      float32x4_t vout3 = vmulq_f32(vloc3, vvar3);

      float32x4_t vloc4 = vld1q_f32(ptr_loc + 12);
      float32x4_t vprior4 = vld1q_f32(ptr_prior + 12);
      float32x4_t vvar4 = vld1q_f32(ptr_var + 12);
      float32x4_t vout4 = vmulq_f32(vloc4, vvar4);

      vst1q_f32(ptr_bbox, vaddq_f32(vout1, vprior1));
      vst1q_f32(ptr_bbox + 4, vaddq_f32(vout2, vprior2));
      vst1q_f32(ptr_bbox + 8, vaddq_f32(vout3, vprior3));
      vst1q_f32(ptr_bbox + 12, vaddq_f32(vout4, vprior4));
    }
    LITE_PARALLEL_END()
    for (int i = cnt * 4; i < num_priors; i++) {
      int idx = i * 4;
      float32x4_t vloc = vld1q_f32(ptr_loc_batch + idx);
      float32x4_t vprior = vld1q_f32(prior_data + idx);
      float32x4_t vvar = vld1q_f32(variance + idx);
      float32x4_t vout = vmulq_f32(vloc, vvar);
      vst1q_f32(ptr_bbox_batch + idx, vaddq_f32(vout, vprior));
    }
  }
}

template <>
void decode_bbox_center_variance_kernel<float>(const int batch_num,
                                               const float* loc_data,
                                               const float* prior_data,
                                               const float* variance,
                                               const int num_priors,
                                               const bool share_location,
                                               const int num_loc_classes,
                                               const int background_label_id,
                                               float* bbox_data) {
  if (!share_location) {
    CHECK_EQ(share_location, true)
        << "ERROR: decode boxes without share_location is unimplemented\n";
    return;
  }

  int cnt = num_priors / 4;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vvar
  float32x4_t vhalf = vdupq_n_f32(0.5f);

  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
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
    LITE_PARALLEL_END()

    LITE_PARALLEL_COMMON_BEGIN(i, tid, num_priors, cnt * 4, 1) {
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
    LITE_PARALLEL_END()
  }
}

template <>
void decode_bbox_center_no_variance_kernel<float>(const int batch_num,
                                                  const float* loc_data,
                                                  const float* prior_data,
                                                  const float* variance,
                                                  const int num_priors,
                                                  const bool share_location,
                                                  const int num_loc_classes,
                                                  const int background_label_id,
                                                  float* bbox_data) {
  if (!share_location) {
    CHECK_EQ(share_location, true)
        << "ERROR: decode boxes without share_location is unimplemented\n";
    return;
  }

  int cnt = num_priors / 4;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vloc   0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! vvar
  float32x4_t vhalf = vdupq_n_f32(0.5f);

  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i * 16;

      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      const float* ptr_var = variance + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4x4_t vprior = vld4q_f32(ptr_prior);
      float32x4x4_t vloc = vld4q_f32(ptr_loc);
      float32x4x4_t vvar = vld4q_f32(ptr_var);
      float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
      float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);
      float32x4_t vprior_cx =
          vmulq_f32(vaddq_f32(vprior.val[0], vprior.val[2]), vhalf);
      float32x4_t vprior_cy =
          vmulq_f32(vaddq_f32(vprior.val[1], vprior.val[3]), vhalf);

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
      vloc.val[2] = vaddq_f32(vdec_bbx_cx, vdec_bbx_w);
      vloc.val[3] = vaddq_f32(vdec_bbx_cy, vdec_bbx_h);

      vst4q_f32(ptr_bbox, vloc);
    }
    LITE_PARALLEL_END()

    LITE_PARALLEL_COMMON_BEGIN(i, tid, num_priors, cnt * 4, 1) {
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
      float decode_bbox_center_x =
          variance[idx] * xmin * prior_width + prior_center_x;
      float decode_bbox_center_y =
          variance[idx + 1] * ymin * prior_height + prior_center_y;
      float decode_bbox_width = expf(variance[idx + 2] * xmax) * prior_width;
      float decode_bbox_height = expf(variance[idx + 3] * ymax) * prior_height;

      ptr_bbox_batch[idx] = decode_bbox_center_x - decode_bbox_width / 2.f;
      ptr_bbox_batch[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
      ptr_bbox_batch[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
      ptr_bbox_batch[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    }
    LITE_PARALLEL_END()
  }
}

template <>
void decode_bbox_corner_size_variance_kernel<float>(
    const int batch_num,
    const float* loc_data,
    const float* prior_data,
    const float* variance,
    const int num_priors,
    const bool share_location,
    const int num_loc_classes,
    const int background_label_id,
    float* bbox_data) {
  if (!share_location) {
    CHECK_EQ(share_location, true)
        << "ERROR: decode boxes without share_location is unimplemented\n";
    return;
  }

  int cnt = num_priors / 4;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! bbx

  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i * 16;

      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      const float* ptr_var = variance + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4x4_t vprior = vld4q_f32(ptr_prior);
      float32x4x4_t vloc = vld4q_f32(ptr_loc);

      float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
      float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);

      float32x4x4_t vbbx;
      vbbx.val[0] = vmulq_f32(vloc.val[0], vprior_width);
      vbbx.val[1] = vmulq_f32(vloc.val[1], vprior_height);
      vbbx.val[2] = vmulq_f32(vloc.val[2], vprior_width);
      vbbx.val[3] = vmulq_f32(vloc.val[3], vprior_height);

      vbbx.val[0] = vaddq_f32(vprior.val[0], vbbx.val[0]);
      vbbx.val[1] = vaddq_f32(vprior.val[1], vbbx.val[1]);
      vbbx.val[2] = vaddq_f32(vprior.val[2], vbbx.val[2]);
      vbbx.val[3] = vaddq_f32(vprior.val[3], vbbx.val[3]);

      vst4q_f32(ptr_bbox, vbbx);
    }
    LITE_PARALLEL_END()

    LITE_PARALLEL_COMMON_BEGIN(i, tid, num_priors, cnt * 4, 1) {
      int idx = i * 4;
      float p_xmin = prior_data[idx];
      float p_ymin = prior_data[idx + 1];
      float p_xmax = prior_data[idx + 2];
      float p_ymax = prior_data[idx + 3];
      float prior_width = p_xmax - p_xmin;
      float prior_height = p_ymax - p_ymin;

      ptr_bbox_batch[idx] = p_xmin + ptr_loc_batch[idx] * prior_width;
      ptr_bbox_batch[idx + 1] = p_ymin + ptr_loc_batch[idx + 1] * prior_height;
      ptr_bbox_batch[idx + 2] = p_xmax + ptr_loc_batch[idx + 2] * prior_width;
      ptr_bbox_batch[idx + 3] = p_ymax + ptr_loc_batch[idx + 3] * prior_height;
    }
    LITE_PARALLEL_END()
  }
}

template <>
void decode_bbox_corner_size_no_variance_kernel<float>(
    const int batch_num,
    const float* loc_data,
    const float* prior_data,
    const float* variance,
    const int num_priors,
    const bool share_location,
    const int num_loc_classes,
    const int background_label_id,
    float* bbox_data) {
  if (!share_location) {
    CHECK_EQ(share_location, true)
        << "ERROR: decode boxes without share_location is unimplemented\n";
    return;
  }

  int cnt = num_priors / 4;
  //! vprior 0: xmin, 1: ymin, 2: xmax, 3: ymax
  //! bbx

  int len_batch = num_priors * 4;

  for (int n = 0; n < batch_num; ++n) {
    const float* ptr_loc_batch = loc_data + n * len_batch;
    float* ptr_bbox_batch = bbox_data + n * len_batch;

    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i * 16;

      const float* ptr_loc = ptr_loc_batch + idx;
      const float* ptr_prior = prior_data + idx;
      const float* ptr_var = variance + idx;
      float* ptr_bbox = ptr_bbox_batch + idx;

      float32x4x4_t vprior = vld4q_f32(ptr_prior);
      float32x4x4_t vloc = vld4q_f32(ptr_loc);

      float32x4_t vprior_width = vsubq_f32(vprior.val[2], vprior.val[0]);
      float32x4_t vprior_height = vsubq_f32(vprior.val[3], vprior.val[1]);

      float32x4x4_t vbbx;
      vbbx.val[0] = vmulq_f32(vloc.val[0], vprior_width);
      vbbx.val[1] = vmulq_f32(vloc.val[1], vprior_height);
      vbbx.val[2] = vmulq_f32(vloc.val[2], vprior_width);
      vbbx.val[3] = vmulq_f32(vloc.val[3], vprior_height);

      vloc = vld4q_f32(ptr_var);
      vbbx.val[0] = vmulq_f32(vbbx.val[0], vloc.val[0]);
      vbbx.val[1] = vmulq_f32(vbbx.val[1], vloc.val[1]);
      vbbx.val[2] = vmulq_f32(vbbx.val[2], vloc.val[2]);
      vbbx.val[3] = vmulq_f32(vbbx.val[3], vloc.val[3]);

      vbbx.val[0] = vaddq_f32(vprior.val[0], vbbx.val[0]);
      vbbx.val[1] = vaddq_f32(vprior.val[1], vbbx.val[1]);
      vbbx.val[2] = vaddq_f32(vprior.val[2], vbbx.val[2]);
      vbbx.val[3] = vaddq_f32(vprior.val[3], vbbx.val[3]);

      vst4q_f32(ptr_bbox, vbbx);
    }
    LITE_PARALLEL_END()

    LITE_PARALLEL_COMMON_BEGIN(i, tid, num_priors, cnt * 4, 1) {
      int idx = i * 4;
      float p_xmin = prior_data[idx];
      float p_ymin = prior_data[idx + 1];
      float p_xmax = prior_data[idx + 2];
      float p_ymax = prior_data[idx + 3];
      float prior_width = p_xmax - p_xmin;
      float prior_height = p_ymax - p_ymin;

      ptr_bbox_batch[idx] =
          p_xmin + ptr_loc_batch[idx] * variance[idx] * prior_width;
      ptr_bbox_batch[idx + 1] =
          p_ymin + ptr_loc_batch[idx + 1] * variance[idx + 1] * prior_height;
      ptr_bbox_batch[idx + 2] =
          p_xmax + ptr_loc_batch[idx + 2] * variance[idx + 2] * prior_width;
      ptr_bbox_batch[idx + 3] =
          p_ymax + ptr_loc_batch[idx + 3] * variance[idx + 3] * prior_height;
    }
    LITE_PARALLEL_END()
  }
}

template <>
void decode_bboxes<float>(const int batch_num,
                          const float* loc_data,
                          const float* prior_data,
                          const std::string code_type,
                          const bool variance_encoded_in_target,
                          const int num_priors,
                          const bool share_location,
                          const int num_loc_classes,
                          const int background_label_id,
                          float* bbox_data) {
  const float* variance_data = prior_data + 4 * num_priors;
  if (code_type == "corner") {
    if (variance_encoded_in_target) {
      decode_bbox_corner_variance_kernel<float>(batch_num,
                                                loc_data,
                                                prior_data,
                                                variance_data,
                                                num_priors,
                                                share_location,
                                                num_loc_classes,
                                                background_label_id,
                                                bbox_data);
    } else {
      decode_bbox_corner_no_variance_kernel<float>(batch_num,
                                                   loc_data,
                                                   prior_data,
                                                   variance_data,
                                                   num_priors,
                                                   share_location,
                                                   num_loc_classes,
                                                   background_label_id,
                                                   bbox_data);
    }
  } else if (code_type == "center_size") {
    if (variance_encoded_in_target) {
      decode_bbox_center_variance_kernel<float>(batch_num,
                                                loc_data,
                                                prior_data,
                                                variance_data,
                                                num_priors,
                                                share_location,
                                                num_loc_classes,
                                                background_label_id,
                                                bbox_data);
    } else {
      decode_bbox_center_no_variance_kernel<float>(batch_num,
                                                   loc_data,
                                                   prior_data,
                                                   variance_data,
                                                   num_priors,
                                                   share_location,
                                                   num_loc_classes,
                                                   background_label_id,
                                                   bbox_data);
    }
  } else if (code_type == "corner_size") {
    if (variance_encoded_in_target) {
      decode_bbox_corner_size_variance_kernel<float>(batch_num,
                                                     loc_data,
                                                     prior_data,
                                                     variance_data,
                                                     num_priors,
                                                     share_location,
                                                     num_loc_classes,
                                                     background_label_id,
                                                     bbox_data);
    } else {
      decode_bbox_corner_size_no_variance_kernel<float>(batch_num,
                                                        loc_data,
                                                        prior_data,
                                                        variance_data,
                                                        num_priors,
                                                        share_location,
                                                        num_loc_classes,
                                                        background_label_id,
                                                        bbox_data);
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
