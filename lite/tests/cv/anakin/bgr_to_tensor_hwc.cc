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

#include "lite/tests/cv/anakin/cv_utils.h"

void bgr_to_tensor_hwc(const uint8_t* bgr,
                       Tensor& output,  // NOLINT
                       int width,
                       int height,
                       float* means,
                       float* scales) {
  int size = width * height;
  float* ptr0 = output.mutable_data<float>();
  float r_means = means[0];
  float g_means = means[1];
  float b_means = means[2];
  float r_scales = scales[0];
  float g_scales = scales[1];
  float b_scales = scales[2];

  int w = width;
  int dim8 = w >> 3;
  int remain = w - (dim8 << 3);

  float32x4_t vrmean = vdupq_n_f32(r_means);
  float32x4_t vgmean = vdupq_n_f32(g_means);
  float32x4_t vbmean = vdupq_n_f32(b_means);
  float32x4_t vrscale = vdupq_n_f32(r_scales);
  float32x4_t vgscale = vdupq_n_f32(g_scales);
  float32x4_t vbscale = vdupq_n_f32(b_scales);

  for (int i = 0; i < height; i++) {
    const uint8_t* ptr_bgr = bgr + i * width * 3;
    float* ptr0_b = ptr0 + i * width;
    float* ptr1_g = ptr0_b + size;
    float* ptr2_r = ptr1_g + size;

    for (int j = 0; j < dim8; j++) {
      uint8x8x3_t vbgr = vld3_u8(ptr_bgr);
      uint8x8_t vb = vbgr.val[0];
      uint8x8_t vg = vbgr.val[1];
      uint8x8_t vr = vbgr.val[2];

      uint16x8_t vb_16 = vmovl_u8(vb);
      uint16x8_t vg_16 = vmovl_u8(vg);
      uint16x8_t vr_16 = vmovl_u8(vr);

      uint32x4_t vb_low_32 = vmovl_u16(vget_low_u16(vb_16));
      uint32x4_t vg_low_32 = vmovl_u16(vget_low_u16(vg_16));
      uint32x4_t vr_low_32 = vmovl_u16(vget_low_u16(vr_16));

      uint32x4_t vb_high_32 = vmovl_u16(vget_high_u16(vb_16));
      uint32x4_t vg_high_32 = vmovl_u16(vget_high_u16(vg_16));
      uint32x4_t vr_high_32 = vmovl_u16(vget_high_u16(vr_16));

      float32x4_t vb_low_f32 = vcvtq_f32_u32(vb_low_32);
      float32x4_t vr_low_f32 = vcvtq_f32_u32(vr_low_32);
      float32x4_t vg_low_f32 = vcvtq_f32_u32(vg_low_32);

      float32x4_t vb_high_f32 = vcvtq_f32_u32(vb_high_32);
      float32x4_t vg_high_f32 = vcvtq_f32_u32(vg_high_32);
      float32x4_t vr_high_f32 = vcvtq_f32_u32(vr_high_32);

      vb_low_f32 = vsubq_f32(vb_low_f32, vbmean);
      vg_low_f32 = vsubq_f32(vg_low_f32, vgmean);
      vr_low_f32 = vsubq_f32(vr_low_f32, vrmean);

      vb_high_f32 = vsubq_f32(vb_high_f32, vbmean);
      vg_high_f32 = vsubq_f32(vg_high_f32, vgmean);
      vr_high_f32 = vsubq_f32(vr_high_f32, vrmean);

      vb_low_f32 = vmulq_f32(vb_low_f32, vbscale);
      vg_low_f32 = vmulq_f32(vg_low_f32, vgscale);
      vr_low_f32 = vmulq_f32(vr_low_f32, vrscale);

      vb_high_f32 = vmulq_f32(vb_high_f32, vbscale);
      vg_high_f32 = vmulq_f32(vg_high_f32, vgscale);
      vr_high_f32 = vmulq_f32(vr_high_f32, vrscale);

      vst1q_f32(ptr0_b, vb_low_f32);
      vst1q_f32(ptr1_g, vg_low_f32);
      vst1q_f32(ptr2_r, vr_low_f32);

      ptr_bgr += 24;

      vst1q_f32(ptr0_b + 4, vb_high_f32);
      vst1q_f32(ptr1_g + 4, vg_high_f32);
      vst1q_f32(ptr2_r + 4, vr_high_f32);

      ptr0_b += 8;
      ptr1_g += 8;
      ptr2_r += 8;
    }

    for (int j = 0; j < remain; j++) {
      *ptr0_b++ = (*ptr_bgr - b_means) * b_scales;  // NOLINT
      ptr_bgr++;
      *ptr1_g++ = (*ptr_bgr - g_means) * g_scales;  // NOLINT
      ptr_bgr++;
      *ptr2_r++ = (*ptr_bgr - r_means) * r_scales;  // NOLINT
      ptr_bgr++;
    }
  }
}
