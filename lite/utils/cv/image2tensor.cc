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

#include "lite/utils/cv/image2tensor.h"
#include <arm_neon.h>
#include "lite/core/parallel_defines.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
void gray_to_tensor(const uint8_t* src,
                    float* output,
                    int width,
                    int height,
                    float* means,
                    float* scales);

void bgr_to_tensor_chw(const uint8_t* src,
                       float* output,
                       int width,
                       int height,
                       float* means,
                       float* scales);

void bgra_to_tensor_chw(const uint8_t* src,
                        float* output,
                        int width,
                        int height,
                        float* means,
                        float* scales);

void bgr_to_tensor_hwc(const uint8_t* src,
                       float* output,
                       int width,
                       int height,
                       float* means,
                       float* scales);

void bgra_to_tensor_hwc(const uint8_t* src,
                        float* output,
                        int width,
                        int height,
                        float* means,
                        float* scales);

/*
  * change image data to tensor data
  * support image format is BGR(RGB) and BGRA(RGBA), Data layout is NHWC and
 * NCHW
  * param src: input image data
  * param dstTensor: output tensor data
  * param srcFormat: input image format, support GRAY, BGR(GRB) and BGRA(RGBA)
  * param srcw: input image width
  * param srch: input image height
  * param layout: output tensor layout，support NHWC and NCHW
  * param means: means of image
  * param scales: scales of image
*/
void Image2Tensor::choose(const uint8_t* src,
                          Tensor* dst,
                          ImageFormat srcFormat,
                          LayoutType layout,
                          int srcw,
                          int srch,
                          float* means,
                          float* scales) {
  float* output = dst->mutable_data<float>();
  if (layout == LayoutType::kNCHW && (srcFormat == BGR || srcFormat == RGB)) {
    impl_ = bgr_to_tensor_chw;
  } else if (layout == LayoutType::kNHWC &&
             (srcFormat == BGR || srcFormat == RGB)) {
    impl_ = bgr_to_tensor_hwc;
  } else if (layout == LayoutType::kNCHW &&
             (srcFormat == BGRA || srcFormat == RGBA)) {
    impl_ = bgra_to_tensor_chw;
  } else if (layout == LayoutType::kNHWC &&
             (srcFormat == BGRA || srcFormat == RGBA)) {
    impl_ = bgra_to_tensor_hwc;
  } else if ((layout == LayoutType::kNHWC || layout == LayoutType::kNCHW) &&
             (srcFormat == GRAY)) {
    impl_ = gray_to_tensor;
  } else {
    printf("this layout: %d or image format: %d not support \n",
           static_cast<int>(layout),
           srcFormat);
    return;
  }
  impl_(src, output, srcw, srch, means, scales);
}

void gray_to_tensor(const uint8_t* src,
                    float* output,
                    int width,
                    int height,
                    float* means,
                    float* scales) {
  int size = width * height;
  float mean_val = means[0];
  float scale_val = scales[0];

  int dim16 = width >> 16;
  int remain = width % 16;

  float32x4_t vmean = vdupq_n_f32(mean_val);
  float32x4_t vscale = vdupq_n_f32(scale_val);
  LITE_PARALLEL_BEGIN(i, tid, height) {
    const uint8_t* din_ptr = src + i * width;
    float* ptr_h = output + i * width;
    int cnt = dim16;
    if (cnt > 0) {
#ifdef __aarch64__
      asm volatile(
          "prfm   pldl1keep, [%[inptr0]]                \n"
          "prfm   pldl1keep, [%[inptr0], #64]   \n"
          "prfm   pldl1keep, [%[inptr0], #128]   \n"
          "prfm   pldl1keep, [%[inptr0], #192]   \n"
          "1:     \n"
          "ld1 {v0.8b}, [%[inptr0]], #8 \n"  // d8 = y0y1y2.."
          "ld1 {v1.8b}, [%[inptr0]], #8 \n"  // d8 = y0y1y2.."
          // 8->16
          "ushll v3.8h, v0.8b, #0  \n"
          "ushll v4.8h, v0.8b, #0  \n"
          // 16->32
          "ushll v6.4s, v3.4h, #0   \n"
          "ushll2 v7.4s, v3.8h, #0   \n"
          "ushll v8.4s, v4.4h, #0   \n"
          "ushll2 v9.4s, v4.8h, #0   \n"
          // int32->fp32
          "ucvtf v12.4s, v6.4s \n"
          "ucvtf v13.4s, v7.4s \n"
          "ucvtf v14.4s, v8.4s \n"
          "ucvtf v15.4s, v9.4s \n"
          // sub -mean
          "fsub v12.4s, v12.4s, %[vmean].4s \n"
          "fsub v13.4s, v13.4s, %[vmean].4s \n"
          "fsub v14.4s, v14.4s, %[vmean].4s \n"
          "fsub v15.4s, v15.4s, %[vmean].4s \n"
          // mul * scale
          "fmul v6.4s, v12.4s, %[vscale].4s \n"
          "fmul v7.4s, v13.4s, %[vscale].4s \n"
          "fmul v8.4s, v14.4s, %[vscale].4s \n"
          "fmul v9.4s, v15.4s, %[vscale].4s \n"
          // store
          "st1 {v6.4s}, [%[outr0]], #16 \n"
          "subs %w[cnt], %w[cnt], #1 \n"
          "st1 {v7.4s}, [%[outr0]], #16 \n"
          "st1 {v8.4s}, [%[outr0]], #16 \n"
          "st1 {v9.4s}, [%[outr0]], #16 \n"
          "bne 1b \n"
          : [inptr0] "+r"(din_ptr), [outr0] "+r"(ptr_h), [cnt] "+r"(cnt)
          : [vmean] "w"(vmean), [vscale] "w"(vscale)
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
          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #64]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #192]                         @ preload a, 64byte\n"
          "1: \n"
          "vld1.8 {d12, d13}, [%[inptr0]]! \n"
          // 8->16
          "vmovl.u8 q8, d12 \n"
          "vmovl.u8 q9, d13 \n"
          // 16->32
          "vmovl.u16 q11, d16 \n"
          "vmovl.u16 q12, d17 \n"
          "vmovl.u16 q13, d18 \n"
          "vmovl.u16 q14, d19 \n"
          // int32->fp32
          "vcvt.f32.u32 q7, q11 \n"
          "vcvt.f32.u32 q8, q12 \n"
          "vcvt.f32.u32 q9, q13 \n"
          "vcvt.f32.u32 q10, q14 \n"
          // sub -mean
          "vsub.f32 q7, q7, %q[vmean] \n"
          "vsub.f32 q8, q8, %q[vmean] \n"
          "vsub.f32 q9, q9, %q[vmean] \n"
          "vsub.f32 q10, q10, %q[vmean] \n"
          // mul *scale
          "vmul.f32 q11, q7, %q[vscale] \n"
          "vmul.f32 q12, q8, %q[vscale] \n"
          "vmul.f32 q13, q9, %q[vscale] \n"
          "vmul.f32 q14, q10, %q[vscale] \n"
          // store
          "vst1.32  {d22 - d23}, [%[outr0]]! \n"
          "subs %[cnt], #1 \n"
          "vst1.32  {d24 - d25}, [%[outr0]]! \n"
          "vst1.32  {d26 - d27}, [%[outr0]]! \n"
          "vst1.32  {d28 - d29}, [%[outr0]]! \n"
          "bne 1b"
          : [inptr0] "+r"(din_ptr), [outr0] "+r"(ptr_h), [cnt] "+r"(cnt)
          : [vmean] "w"(vmean), [vscale] "w"(vscale)
          : "cc",
            "memory",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14");
#endif
    }
    for (int j = 0; j < remain; j++) {
      *ptr_h++ = (*din_ptr - mean_val) * scale_val;
      din_ptr++;
    }
  }
  LITE_PARALLEL_END();
}

void bgr_to_tensor_chw(const uint8_t* src,
                       float* output,
                       int width,
                       int height,
                       float* means,
                       float* scales) {
  int size = width * height;
  float b_means = means[0];
  float g_means = means[1];
  float r_means = means[2];
  float b_scales = scales[0];
  float g_scales = scales[1];
  float r_scales = scales[2];

  float* ptr_b = output;
  float* ptr_g = ptr_b + size;
  float* ptr_r = ptr_g + size;

  int dim8 = width >> 3;
  int remain = width % 8;

  float32x4_t vbmean = vdupq_n_f32(b_means);
  float32x4_t vgmean = vdupq_n_f32(g_means);
  float32x4_t vrmean = vdupq_n_f32(r_means);
  float32x4_t vbscale = vdupq_n_f32(b_scales);
  float32x4_t vgscale = vdupq_n_f32(g_scales);
  float32x4_t vrscale = vdupq_n_f32(r_scales);
  LITE_PARALLEL_BEGIN(i, tid, height) {
    const uint8_t* din_ptr = src + i * 3 * width;
    float* ptr_b_h = ptr_b + i * width;
    float* ptr_g_h = ptr_g + i * width;
    float* ptr_r_h = ptr_r + i * width;
    int cnt = dim8;
    if (cnt > 0) {
#ifdef __aarch64__
      asm volatile(
          "prfm   pldl1keep, [%[inptr0]]                \n"
          "prfm   pldl1keep, [%[inptr0], #64]   \n"
          "prfm   pldl1keep, [%[inptr0], #128]   \n"
          "prfm   pldl1keep, [%[inptr0], #192]   \n"
          "1:     \n"
          "ld3 {v0.8b, v1.8b, v2.8b}, [%[inptr0]], #24 \n"  // d8 = y0y3y6y9..
                                                            // d9 = y1y4y7..."
          // 8->16
          "ushll v3.8h, v0.8b, #0  \n"
          "ushll v4.8h, v1.8b, #0  \n"
          "ushll v5.8h, v2.8b, #0  \n"
          // 16->32
          "ushll v6.4s, v3.4h, #0   \n"
          "ushll2 v7.4s, v3.8h, #0   \n"
          "ushll v8.4s, v4.4h, #0   \n"
          "ushll2 v9.4s, v4.8h, #0   \n"
          "ushll v10.4s, v5.4h, #0  \n"
          "ushll2 v11.4s, v5.8h, #0   \n"
          // int32->fp32
          "ucvtf v12.4s, v6.4s \n"
          "ucvtf v13.4s, v7.4s \n"
          "ucvtf v14.4s, v8.4s \n"
          "ucvtf v15.4s, v9.4s \n"
          "ucvtf v16.4s, v10.4s \n"
          "ucvtf v17.4s, v11.4s \n"
          // sub -mean
          "fsub v12.4s, v12.4s, %[vbmean].4s \n"
          "fsub v13.4s, v13.4s, %[vbmean].4s \n"
          "fsub v14.4s, v14.4s, %[vgmean].4s \n"
          "fsub v15.4s, v15.4s, %[vgmean].4s \n"
          "fsub v16.4s, v16.4s, %[vrmean].4s \n"
          "fsub v17.4s, v17.4s, %[vrmean].4s \n"
          // mul * scale
          "fmul v6.4s, v12.4s, %[vbscale].4s \n"
          "fmul v7.4s, v13.4s, %[vbscale].4s \n"
          "fmul v8.4s, v14.4s, %[vgscale].4s \n"
          "fmul v9.4s, v15.4s, %[vgscale].4s \n"
          "fmul v10.4s, v16.4s, %[vrscale].4s \n"
          "fmul v11.4s, v17.4s, %[vrscale].4s \n"
          // store
          "st1 {v6.4s}, [%[outr0]], #16 \n"
          "st1 {v8.4s}, [%[outr1]], #16 \n"
          "st1 {v10.4s}, [%[outr2]], #16 \n"
          "subs %w[cnt], %w[cnt], #1 \n"
          "st1 {v7.4s}, [%[outr0]], #16 \n"
          "st1 {v9.4s}, [%[outr1]], #16 \n"
          "st1 {v11.4s}, [%[outr2]], #16 \n"
          "bne 1b \n"
          : [inptr0] "+r"(din_ptr),
            [outr0] "+r"(ptr_b_h),
            [outr1] "+r"(ptr_g_h),
            [outr2] "+r"(ptr_r_h),
            [cnt] "+r"(cnt)
          : [vbmean] "w"(vbmean),
            [vgmean] "w"(vgmean),
            [vrmean] "w"(vrmean),
            [vbscale] "w"(vbscale),
            [vgscale] "w"(vgscale),
            [vrscale] "w"(vrscale)
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
            "v15",
            "v16",
            "v17",
            "v18",
            "v19",
            "v20");
#else
      asm volatile(
          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #64]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #192]                         @ preload a, 64byte\n"
          "1: \n"
          "vld3.8 {d12, d13, d14}, [%[inptr0]]! \n"
          // 8->16
          "vmovl.u8 q8, d12 \n"
          "vmovl.u8 q9, d13 \n"
          "vmovl.u8 q10, d14 \n"
          // 16->32
          "vmovl.u16 q11, d16 \n"
          "vmovl.u16 q12, d17 \n"
          "vmovl.u16 q13, d18 \n"
          "vmovl.u16 q14, d19 \n"
          "vmovl.u16 q15, d20 \n"
          "vmovl.u16 q6, d21 \n"
          // int32->fp32
          "vcvt.f32.u32 q7, q11 \n"
          "vcvt.f32.u32 q8, q12 \n"
          "vcvt.f32.u32 q9, q13 \n"
          "vcvt.f32.u32 q10, q14 \n"
          "vcvt.f32.u32 q11, q15 \n"
          "vcvt.f32.u32 q12, q6 \n"
          // sub -mean
          "vsub.f32 q7, q7, %q[vbmean] \n"
          "vsub.f32 q8, q8, %q[vbmean] \n"
          "vsub.f32 q9, q9, %q[vgmean] \n"
          "vsub.f32 q10, q10, %q[vgmean] \n"
          "vsub.f32 q11, q11, %q[vrmean] \n"
          "vsub.f32 q12, q12, %q[vrmean] \n"
          // mul *scale
          "vmul.f32 q13, q7, %q[vbscale] \n"
          "vmul.f32 q14, q8, %q[vbscale] \n"
          "vmul.f32 q15, q9, %q[vgscale] \n"
          "vmul.f32 q6, q10, %q[vgscale] \n"
          "vmul.f32 q7, q11, %q[vrscale] \n"
          "vmul.f32 q8, q12, %q[vrscale] \n"
          // store
          "vst1.32  {d26 - d27}, [%[outr0]]! \n"
          "vst1.32  {d30 - d31}, [%[outr1]]! \n"
          "vst1.32  {d14 - d15}, [%[outr2]]! \n"
          "subs %[cnt], #1 \n"
          "vst1.32  {d28 - d29}, [%[outr0]]! \n"
          "vst1.32  {d12 - d13}, [%[outr1]]! \n"
          "vst1.32  {d16 - d17}, [%[outr2]]! \n"
          "bne 1b"
          : [inptr0] "+r"(din_ptr),
            [outr0] "+r"(ptr_b_h),
            [outr1] "+r"(ptr_g_h),
            [outr2] "+r"(ptr_r_h),
            [cnt] "+r"(cnt)
          : [vbmean] "w"(vbmean),
            [vgmean] "w"(vgmean),
            [vrmean] "w"(vrmean),
            [vbscale] "w"(vbscale),
            [vgscale] "w"(vgscale),
            [vrscale] "w"(vrscale)
          : "cc",
            "memory",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15");
#endif
    }
    for (int j = 0; j < remain; j++) {
      *ptr_b_h++ = (*din_ptr - b_means) * b_scales;
      din_ptr++;
      *ptr_g_h++ = (*din_ptr - g_means) * g_scales;
      din_ptr++;
      *ptr_r_h++ = (*din_ptr - r_means) * r_scales;
      din_ptr++;
    }
  }
  LITE_PARALLEL_END();
}

void bgra_to_tensor_chw(const uint8_t* src,
                        float* output,
                        int width,
                        int height,
                        float* means,
                        float* scales) {
  int size = width * height;
  float b_means = means[0];
  float g_means = means[1];
  float r_means = means[2];
  float b_scales = scales[0];
  float g_scales = scales[1];
  float r_scales = scales[2];

  float* ptr_b = output;
  float* ptr_g = ptr_b + size;
  float* ptr_r = ptr_g + size;

  int dim8 = width >> 3;
  int remain = width % 8;

  float32x4_t vbmean = vdupq_n_f32(b_means);
  float32x4_t vgmean = vdupq_n_f32(g_means);
  float32x4_t vrmean = vdupq_n_f32(r_means);
  float32x4_t vbscale = vdupq_n_f32(b_scales);
  float32x4_t vgscale = vdupq_n_f32(g_scales);
  float32x4_t vrscale = vdupq_n_f32(r_scales);
  LITE_PARALLEL_BEGIN(i, tid, height) {
    const uint8_t* din_ptr = src + i * 4 * width;
    float* ptr_b_h = ptr_b + i * width;
    float* ptr_g_h = ptr_g + i * width;
    float* ptr_r_h = ptr_r + i * width;

    for (int j = 0; j < dim8; j++) {
      uint8x8x4_t v_bgr = vld4_u8(din_ptr);

      uint16x8_t vb_16 = vmovl_u8(v_bgr.val[0]);
      uint16x8_t vg_16 = vmovl_u8(v_bgr.val[1]);
      uint16x8_t vr_16 = vmovl_u8(v_bgr.val[2]);

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

      vst1q_f32(ptr_b_h, vb_low_f32);
      vst1q_f32(ptr_g_h, vg_low_f32);
      vst1q_f32(ptr_r_h, vr_low_f32);

      din_ptr += 32;

      vst1q_f32(ptr_b_h + 4, vb_high_f32);
      vst1q_f32(ptr_g_h + 4, vg_high_f32);
      vst1q_f32(ptr_r_h + 4, vr_high_f32);

      ptr_b_h += 8;
      ptr_g_h += 8;
      ptr_r_h += 8;
    }

    for (int j = 0; j < remain; j++) {
      *ptr_b_h++ = (*din_ptr - b_means) * b_scales;
      din_ptr++;
      *ptr_g_h++ = (*din_ptr - g_means) * g_scales;
      din_ptr++;
      *ptr_r_h++ = (*din_ptr - r_means) * r_scales;
      din_ptr++;
      din_ptr++;  // a
    }
  }
  LITE_PARALLEL_END();
}

void bgr_to_tensor_hwc(const uint8_t* src,
                       float* output,
                       int width,
                       int height,
                       float* means,
                       float* scales) {
  int size = width * height;
  float b_means = means[0];
  float g_means = means[1];
  float r_means = means[2];
  float b_scales = scales[0];
  float g_scales = scales[1];
  float r_scales = scales[2];

  float* dout = output;

  int dim8 = width >> 3;
  int remain = width % 8;

  float32x4_t vbmean = vdupq_n_f32(b_means);
  float32x4_t vgmean = vdupq_n_f32(g_means);
  float32x4_t vrmean = vdupq_n_f32(r_means);
  float32x4_t vbscale = vdupq_n_f32(b_scales);
  float32x4_t vgscale = vdupq_n_f32(g_scales);
  float32x4_t vrscale = vdupq_n_f32(r_scales);
  LITE_PARALLEL_BEGIN(i, tid, height) {
    const uint8_t* din_ptr = src + i * 3 * width;
    float* dout_ptr = dout + i * 3 * width;

    for (int j = 0; j < dim8; j++) {
      uint8x8x3_t v_bgr = vld3_u8(din_ptr);

      uint16x8_t vb_16 = vmovl_u8(v_bgr.val[0]);
      uint16x8_t vg_16 = vmovl_u8(v_bgr.val[1]);
      uint16x8_t vr_16 = vmovl_u8(v_bgr.val[2]);

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

      float32x4x3_t val;
      val.val[0] = vb_low_f32;
      val.val[1] = vg_low_f32;
      val.val[2] = vr_low_f32;

      vst3q_f32(dout_ptr, val);

      din_ptr += 24;
      dout_ptr += 12;

      val.val[0] = vb_high_f32;
      val.val[1] = vg_high_f32;
      val.val[2] = vr_high_f32;

      vst3q_f32(dout_ptr, val);

      dout_ptr += 12;
    }

    for (int j = 0; j < remain; j++) {
      *dout_ptr++ = (*din_ptr - b_means) * b_scales;
      din_ptr++;
      *dout_ptr++ = (*din_ptr - g_means) * g_scales;
      din_ptr++;
      *dout_ptr++ = (*din_ptr - r_means) * r_scales;
      din_ptr++;
    }
  }
  LITE_PARALLEL_END();
}

void bgra_to_tensor_hwc(const uint8_t* src,
                        float* output,
                        int width,
                        int height,
                        float* means,
                        float* scales) {
  int size = width * height;
  float b_means = means[0];
  float g_means = means[1];
  float r_means = means[2];
  float b_scales = scales[0];
  float g_scales = scales[1];
  float r_scales = scales[2];

  float* dout = output;

  int dim8 = width >> 3;
  int remain = width % 8;

  float32x4_t vbmean = vdupq_n_f32(b_means);
  float32x4_t vgmean = vdupq_n_f32(g_means);
  float32x4_t vrmean = vdupq_n_f32(r_means);
  float32x4_t vbscale = vdupq_n_f32(b_scales);
  float32x4_t vgscale = vdupq_n_f32(g_scales);
  float32x4_t vrscale = vdupq_n_f32(r_scales);
  LITE_PARALLEL_BEGIN(i, tid, height) {
    const uint8_t* din_ptr = src + i * 4 * width;
    float* dout_ptr = dout + i * 3 * width;

    for (int j = 0; j < dim8; j++) {
      uint8x8x4_t v_bgr = vld4_u8(din_ptr);

      uint16x8_t vb_16 = vmovl_u8(v_bgr.val[0]);
      uint16x8_t vg_16 = vmovl_u8(v_bgr.val[1]);
      uint16x8_t vr_16 = vmovl_u8(v_bgr.val[2]);
      // uint16x8_t va_16 = vmovl_u8(v_bgr.val[3]);

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

      float32x4x3_t val;
      val.val[0] = vb_low_f32;
      val.val[1] = vg_low_f32;
      val.val[2] = vr_low_f32;
      // val.val[3] = num_a;

      vst3q_f32(dout_ptr, val);

      din_ptr += 32;
      dout_ptr += 12;

      val.val[0] = vb_high_f32;
      val.val[1] = vg_high_f32;
      val.val[2] = vr_high_f32;

      vst3q_f32(dout_ptr, val);

      dout_ptr += 12;
    }

    for (int j = 0; j < remain; j++) {
      *dout_ptr++ = (*din_ptr - b_means) * b_scales;
      din_ptr++;
      *dout_ptr++ = (*din_ptr - g_means) * g_scales;
      din_ptr++;
      *dout_ptr++ = (*din_ptr - r_means) * r_scales;
      din_ptr++;
      din_ptr++;  // a
      // *dout_ptr++ = 255;
    }
  }
  LITE_PARALLEL_END();
}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
