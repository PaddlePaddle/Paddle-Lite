// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/fp16/conv_transpose_depthwise_fp16.h"
#include <arm_neon.h>
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
template <>
void conv_transpose_depthwise_s1_fp16<float16_t>(const float16_t* dst,
                                                 const float16_t* weights,
                                                 const int channels,
                                                 const int height,
                                                 const int width,
                                                 const int kernel_h,
                                                 const int kernel_w,
                                                 const int pad_h0,
                                                 const int pad_h1,
                                                 const int pad_w0,
                                                 const int pad_w1,
                                                 const int dilation_h,
                                                 const int dilation_w,
                                                 float16_t* src,
                                                 ARMContext* ctx) {
  memset(src, 0, height * width * channels * sizeof(float16_t));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) + 1;
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, width * sizeof(float16_t));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel_h * kernel_w;
  const int16_t width_s16 = static_cast<int16_t>(width);
  LITE_PARALLEL_BEGIN(c, tid, channels) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel_h; ky++) {
      int weight_y = ky * kernel_w;
      for (int kx = 0; kx < kernel_w; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float16_t* weight_addr = weights + weight_offset;
        for (int ih = -pad_h0 + ky * dilation_h, oh = 0; oh < output_h;
             ih += 4, oh += 4) {
          int src_y = ih * width;
          int dst_y = oh * output_w;
          bool boundary_y0 = ((ih >= 0) && (ih < height)) && (oh < output_h);
          bool boundary_y1 =
              ((ih + 1) >= 0) && ((ih + 1) < height) && ((oh + 1) < output_h);
          bool boundary_y2 =
              ((ih + 2) >= 0) && ((ih + 2) < height) && ((oh + 2) < output_h);
          bool boundary_y3 =
              ((ih + 3) >= 0) && ((ih + 3) < height) && ((oh + 3) < output_h);
          float16_t* src_addr_h0 =
              boundary_y0 ? (src + src_z + src_y) : zero_ptr;
          float16_t* src_addr_h1 =
              boundary_y1 ? (src + src_z + width + src_y) : zero_ptr;
          float16_t* src_addr_h2 =
              boundary_y2 ? (src + src_z + width * 2 + src_y) : zero_ptr;
          float16_t* src_addr_h3 =
              boundary_y3 ? (src + src_z + width * 3 + src_y) : zero_ptr;
          int16_t iw = -pad_w0 + kx * dilation_w;
          int16_t i = 0;
          if (iw < 0) {
            i = -iw;
            iw = 0;
          }
          for (; i + 7 < output_w && iw + 7 < width; i += 8, iw += 8) {
            int dst_offset = dst_z + dst_y + i;
            const float16_t* dst_addr = dst + dst_offset;

            if (boundary_y0) {
              float16x8_t src_v0 = vfmaq_f16(vld1q_f16(src_addr_h0 + iw),
                                             vld1q_f16(dst_addr),
                                             vld1q_dup_f16(weight_addr));
              vst1q_f16(src_addr_h0 + iw, src_v0);
            }

            if (boundary_y1) {
              float16x8_t src_v2 = vfmaq_f16(vld1q_f16(src_addr_h1 + iw),
                                             vld1q_f16(dst_addr + output_w),
                                             vld1q_dup_f16(weight_addr));
              vst1q_f16(src_addr_h1 + iw, src_v2);
            }

            if (boundary_y2) {
              float16x8_t src_v4 = vfmaq_f16(vld1q_f16(src_addr_h2 + iw),
                                             vld1q_f16(dst_addr + output_w * 2),
                                             vld1q_dup_f16(weight_addr));
              vst1q_f16(src_addr_h2 + iw, src_v4);
            }

            if (boundary_y3) {
              float16x8_t src_v6 = vfmaq_f16(vld1q_f16(src_addr_h3 + iw),
                                             vld1q_f16(dst_addr + output_w * 3),
                                             vld1q_dup_f16(weight_addr));
              vst1q_f16(src_addr_h3 + iw, src_v6);
            }
          }
          for (; i + 3 < output_w && iw + 3 < width; i += 4, iw += 4) {
            int dst_offset = dst_z + dst_y + i;
            const float16_t* dst_addr = dst + dst_offset;

            if (boundary_y0) {
              float16x4_t src_v0 = vfma_f16(vld1_f16(src_addr_h0 + iw),
                                            vld1_f16(dst_addr),
                                            vld1_dup_f16(weight_addr));
              vst1_f16(src_addr_h0 + iw, src_v0);
            }

            if (boundary_y1) {
              float16x4_t src_v2 = vfma_f16(vld1_f16(src_addr_h1 + iw),
                                            vld1_f16(dst_addr + output_w),
                                            vld1_dup_f16(weight_addr));
              vst1_f16(src_addr_h1 + iw, src_v2);
            }

            if (boundary_y2) {
              float16x4_t src_v4 = vfma_f16(vld1_f16(src_addr_h2 + iw),
                                            vld1_f16(dst_addr + output_w * 2),
                                            vld1_dup_f16(weight_addr));
              vst1_f16(src_addr_h2 + iw, src_v4);
            }

            if (boundary_y3) {
              float16x4_t src_v6 = vfma_f16(vld1_f16(src_addr_h3 + iw),
                                            vld1_f16(dst_addr + output_w * 3),
                                            vld1_dup_f16(weight_addr));
              vst1_f16(src_addr_h3 + iw, src_v6);
            }
          }
          for (; i < output_w; i++, iw++) {
            bool boundary_x = ((iw >= 0) && (iw < width));
            int src_offset = src_z + src_y + iw;
            int dst_offset = dst_z + dst_y + i;
            if (boundary_x && boundary_y0)
              src[src_offset] += dst[dst_offset] * weights[weight_offset];
            if (boundary_x && boundary_y1)
              src[src_offset + width] +=
                  dst[dst_offset + output_w] * weights[weight_offset];
            if (boundary_x && boundary_y2)
              src[src_offset + width * 2] +=
                  dst[dst_offset + output_w * 2] * weights[weight_offset];
            if (boundary_x && boundary_y3)
              src[src_offset + width * 3] +=
                  dst[dst_offset + output_w * 3] * weights[weight_offset];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

template <>
void conv_transpose_depthwise_s2_fp16<float16_t>(const float16_t* dst,
                                                 const float16_t* weights,
                                                 const int channels,
                                                 const int height,
                                                 const int width,
                                                 const int kernel_h,
                                                 const int kernel_w,
                                                 const int pad_h0,
                                                 const int pad_h1,
                                                 const int pad_w0,
                                                 const int pad_w1,
                                                 const int dilation_h,
                                                 const int dilation_w,
                                                 float16_t* src,
                                                 ARMContext* ctx) {
  memset(src, 0, height * width * channels * sizeof(float16_t));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) / 2 + 1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / 2 + 1;
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, width * sizeof(float16_t));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel_h * kernel_w;
  const int16_t width_s16 = static_cast<int16_t>(width);
  LITE_PARALLEL_BEGIN(c, tid, channels) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel_h; ky++) {
      int weight_y = ky * kernel_w;
      for (int kx = 0; kx < kernel_w; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float16_t* weight_addr = weights + weight_offset;
        for (int ih = -pad_h0 + ky * dilation_h, oh = 0; oh < output_h;
             ih += 8, oh += 4) {
          int src_y = ih * width;
          int dst_y = oh * output_w;
          bool boundary_y0 = ((ih >= 0) && (ih < height)) && (oh < output_h);
          bool boundary_y1 =
              ((ih + 2) >= 0) && ((ih + 2) < height) && ((oh + 1) < output_h);
          bool boundary_y2 =
              ((ih + 4) >= 0) && ((ih + 4) < height) && ((oh + 2) < output_h);
          bool boundary_y3 =
              ((ih + 6) >= 0) && ((ih + 6) < height) && ((oh + 3) < output_h);
          float16_t* src_addr_h0 =
              boundary_y0 ? (src + src_z + src_y) : zero_ptr;
          float16_t* src_addr_h1 =
              boundary_y1 ? (src + src_z + width * 2 + src_y) : zero_ptr;
          float16_t* src_addr_h2 =
              boundary_y2 ? (src + src_z + width * 4 + src_y) : zero_ptr;
          float16_t* src_addr_h3 =
              boundary_y3 ? (src + src_z + width * 6 + src_y) : zero_ptr;
          int iw = -pad_w0 + kx * dilation_w;
          int i = 0;
          if (iw < 0) {
            i = ((-iw % 2) - iw) / 2;
            iw = -iw % 2;
          }
          for (; i + 7 < output_w && iw + 14 < width; i += 8, iw += 16) {
            int dst_offset = dst_z + dst_y + i;
            const float16_t* dst_addr = dst + dst_offset;

            if (boundary_y0) {
              float16x8x2_t src_vv0 = vld2q_f16(src_addr_h0 + iw);
              src_vv0.val[0] = vfmaq_f16(src_vv0.val[0],
                                         vld1q_f16(dst_addr),
                                         vld1q_dup_f16(weight_addr));
              vst2q_f16(src_addr_h0 + iw, src_vv0);
            }

            if (boundary_y1) {
              float16x8x2_t src_vv2 = vld2q_f16(src_addr_h1 + iw);
              src_vv2.val[0] = vfmaq_f16(src_vv2.val[0],
                                         vld1q_f16(dst_addr + output_w),
                                         vld1q_dup_f16(weight_addr));
              vst2q_f16(src_addr_h1 + iw, src_vv2);
            }

            if (boundary_y2) {
              float16x8x2_t src_vv4 = vld2q_f16(src_addr_h2 + iw);
              src_vv4.val[0] = vfmaq_f16(src_vv4.val[0],
                                         vld1q_f16(dst_addr + output_w * 2),
                                         vld1q_dup_f16(weight_addr));
              vst2q_f16(src_addr_h2 + iw, src_vv4);
            }

            if (boundary_y3) {
              float16x8x2_t src_vv6 = vld2q_f16(src_addr_h3 + iw);
              src_vv6.val[0] = vfmaq_f16(src_vv6.val[0],
                                         vld1q_f16(dst_addr + output_w * 3),
                                         vld1q_dup_f16(weight_addr));
              vst2q_f16(src_addr_h3 + iw, src_vv6);
            }
          }
          for (; i + 3 < output_w && iw + 6 < width; i += 4, iw += 8) {
            int dst_offset = dst_z + dst_y + i;
            const float16_t* dst_addr = dst + dst_offset;

            if (boundary_y0) {
              float16x4x2_t src_vv0 = vld2_f16(src_addr_h0 + iw);
              src_vv0.val[0] = vfma_f16(src_vv0.val[0],
                                        vld1_f16(dst_addr),
                                        vld1_dup_f16(weight_addr));
              vst2_f16(src_addr_h0 + iw, src_vv0);
            }

            if (boundary_y1) {
              float16x4x2_t src_vv1 = vld2_f16(src_addr_h1 + iw);
              src_vv1.val[0] = vfma_f16(src_vv1.val[0],
                                        vld1_f16(dst_addr + output_w),
                                        vld1_dup_f16(weight_addr));
              vst2_f16(src_addr_h1 + iw, src_vv1);
            }

            if (boundary_y2) {
              float16x4x2_t src_vv2 = vld2_f16(src_addr_h2 + iw);
              src_vv2.val[0] = vfma_f16(src_vv2.val[0],
                                        vld1_f16(dst_addr + output_w * 2),
                                        vld1_dup_f16(weight_addr));
              vst2_f16(src_addr_h2 + iw, src_vv2);
            }

            if (boundary_y3) {
              float16x4x2_t src_vv3 = vld2_f16(src_addr_h3 + iw);
              src_vv3.val[0] = vfma_f16(src_vv3.val[0],
                                        vld1_f16(dst_addr + output_w * 3),
                                        vld1_dup_f16(weight_addr));
              vst2_f16(src_addr_h3 + iw, src_vv3);
            }
          }
          for (; i < output_w; i++, iw += 2) {
            bool boundary_x = ((iw >= 0) && (iw < width));
            int src_offset = src_z + src_y + iw;
            int dst_offset = dst_z + dst_y + i;

            if (boundary_x && boundary_y0)
              src[src_offset] += dst[dst_offset] * weights[weight_offset];
            if (boundary_x && boundary_y1)
              src[src_offset + width * 2] +=
                  dst[dst_offset + output_w] * weights[weight_offset];
            if (boundary_x && boundary_y2)
              src[src_offset + width * 4] +=
                  dst[dst_offset + output_w * 2] * weights[weight_offset];
            if (boundary_x && boundary_y3)
              src[src_offset + width * 6] +=
                  dst[dst_offset + output_w * 3] * weights[weight_offset];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
