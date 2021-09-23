/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/conv2d_transpose.h"
#include <string.h>
#include "lite/backends/x86/math/avx/avx_mathfuns.h"

#ifdef __AVX__
#include <immintrin.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

static bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void col2im(const float* data_col,
            const int channels,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h0,
            const int pad_h1,
            const int pad_w0,
            const int pad_w1,
            const int stride_h,
            const int stride_w,
            const int dilation_h,
            const int dilation_w,
            float* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h0 + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w0 + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void conv_transpose_depthwise_s1(const float* dst,
                                 const float* weights,
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
                                 float* src,
                                 X86Context* ctx) {
  memset(src, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) + 1;
  float* zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kX86), width * sizeof(float)));
  memset(zero_ptr, 0, width * sizeof(float));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel_h * kernel_w;

#ifdef __AVX__
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_width = _mm256_set1_ps(width * 1.0f);
#endif
#ifdef __SSE4_1__  // blendv need 4.1
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_width_128 = _mm_set1_ps(width * 1.0f);
#endif

  for (int c = 0; c < channels; c++) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel_h; ky++) {
      int weight_y = ky * kernel_w;
      for (int kx = 0; kx < kernel_w; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float* weight_addr = weights + weight_offset;
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
          float* src_addr_h0 = boundary_y0 ? (src + src_z + src_y) : zero_ptr;
          float* src_addr_h1 =
              boundary_y1 ? (src + src_z + width + src_y) : zero_ptr;
          float* src_addr_h2 =
              boundary_y2 ? (src + src_z + width * 2 + src_y) : zero_ptr;
          float* src_addr_h3 =
              boundary_y3 ? (src + src_z + width * 3 + src_y) : zero_ptr;
          int iw = -pad_w0 + kx * dilation_w;
          int i = 0;

#ifdef __AVX__
          for (; i + 7 < output_w; i += 8, iw += 8) {
            int dst_offset = dst_z + dst_y + i;
            const float* dst_addr = dst + dst_offset;
            const float iw_data[8] = {iw + 0.f,
                                      iw + 1.f,
                                      iw + 2.f,
                                      iw + 3.f,
                                      iw + 4.f,
                                      iw + 5.f,
                                      iw + 6.f,
                                      iw + 7.f};
            // select weight
            __m256 vec_iw = _mm256_loadu_ps(&iw_data[0]);
            __m256 vec_mask = _mm256_and_ps(
                _mm256_cmp_ps(vec_iw, vec_zero, 13),
                _mm256_cmp_ps(vec_iw, vec_width, 1));  // GE:13  LT:1
            __m256 vec_weight = _mm256_set1_ps(weight_addr[0]);
            vec_weight = _mm256_blendv_ps(vec_zero, vec_weight, vec_mask);

            // compute 4 lines
            __m256 vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr),
                                             vec_weight,
                                             _mm256_loadu_ps(src_addr_h0 + iw));
            _mm256_storeu_ps(src_addr_h0 + iw, vec_dst);

            vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr + output_w),
                                      vec_weight,
                                      _mm256_loadu_ps(src_addr_h1 + iw));
            _mm256_storeu_ps(src_addr_h1 + iw, vec_dst);

            vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr + 2 * output_w),
                                      vec_weight,
                                      _mm256_loadu_ps(src_addr_h2 + iw));
            _mm256_storeu_ps(src_addr_h2 + iw, vec_dst);

            vec_dst = _mm256_fmadd_ps(_mm256_loadu_ps(dst_addr + 3 * output_w),
                                      vec_weight,
                                      _mm256_loadu_ps(src_addr_h3 + iw));
            _mm256_storeu_ps(src_addr_h3 + iw, vec_dst);
          }
#endif
#ifdef __SSE4_1__
          for (; i + 3 < output_w; i += 4, iw += 4) {
            int dst_offset = dst_z + dst_y + i;
            const float* dst_addr = dst + dst_offset;
            const float iw_data[4] = {iw + 0.f, iw + 1.f, iw + 2.f, iw + 3.f};
            // select weight
            __m128 vec_iw_128 = _mm_loadu_ps(&iw_data[0]);
            __m128 vec_mask_128 =
                _mm_and_ps(_mm_cmpge_ps(vec_iw_128, vec_zero_128),
                           _mm_cmplt_ps(vec_iw_128, vec_width_128));
            __m128 vec_weight_128 = _mm_set1_ps(weight_addr[0]);
            vec_weight_128 =
                _mm_blendv_ps(vec_zero_128, vec_weight_128, vec_mask_128);

            // compute 4 lines
            __m128 vec_dst_128 =
                _mm_add_ps(_mm_mul_ps(vec_weight_128, _mm_loadu_ps(dst_addr)),
                           _mm_loadu_ps(src_addr_h0 + iw));
            _mm_storeu_ps(src_addr_h0 + iw, vec_dst_128);

            vec_dst_128 = _mm_add_ps(
                _mm_mul_ps(vec_weight_128, _mm_loadu_ps(dst_addr + output_w)),
                _mm_loadu_ps(src_addr_h1 + iw));
            _mm_storeu_ps(src_addr_h1 + iw, vec_dst_128);

            vec_dst_128 =
                _mm_add_ps(_mm_mul_ps(vec_weight_128,
                                      _mm_loadu_ps(dst_addr + 2 * output_w)),
                           _mm_loadu_ps(src_addr_h2 + iw));
            _mm_storeu_ps(src_addr_h2 + iw, vec_dst_128);

            vec_dst_128 =
                _mm_add_ps(_mm_mul_ps(vec_weight_128,
                                      _mm_loadu_ps(dst_addr + 3 * output_w)),
                           _mm_loadu_ps(src_addr_h3 + iw));
            _mm_storeu_ps(src_addr_h3 + iw, vec_dst_128);
          }
#endif
          for (; i < output_w; i++, iw++) {
            bool boundary_x = ((iw >= 0) && (iw < width));
            int src_offset = src_z + src_y + iw;
            int dst_offset = dst_z + dst_y + i;
            src[src_offset] += (boundary_x) * (boundary_y0)*dst[dst_offset] *
                               weights[weight_offset];
            src[src_offset + width] +=
                (boundary_x) * (boundary_y1)*dst[dst_offset + output_w] *
                weights[weight_offset];
            src[src_offset + width * 2] +=
                (boundary_x) * (boundary_y2)*dst[dst_offset + output_w * 2] *
                weights[weight_offset];
            src[src_offset + width * 3] +=
                (boundary_x) * (boundary_y3)*dst[dst_offset + output_w * 3] *
                weights[weight_offset];
          }
        }
      }
    }
  }
  TargetFree(TARGET(kX86), zero_ptr);
}

void conv_transpose_depthwise_s2(const float* dst,
                                 const float* weights,
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
                                 float* src,
                                 X86Context* ctx) {
  memset(src, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + pad_h0 + pad_h1 - (dilation_h * (kernel_h - 1) + 1)) / 2 + 1;
  const int output_w =
      (width + pad_w0 + pad_w1 - (dilation_w * (kernel_w - 1) + 1)) / 2 + 1;
  float* zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kX86), width * sizeof(float)));
  memset(zero_ptr, 0, width * sizeof(float));
  const int ic_plane_size = height * width;
  const int oc_plane_size = output_h * output_w;
  const int rr_plane_size = kernel_h * kernel_w;

#ifdef __AVX2__
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_width = _mm256_set1_ps(width * 1.0f);
  const int mask_store[8] = {-1, 0, -1, 0, -1, 0, -1, 0};
  __m256i vec_store_mask = _mm256_loadu_si256((const __m256i*)&mask_store[0]);
#endif
#ifdef __SSE4_1__
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_width_128 = _mm_set1_ps(width * 1.0f);
#endif

  for (int c = 0; c < channels; c++) {
    int dst_z = c * oc_plane_size;
    int weight_z = c * rr_plane_size;
    int src_z = c * ic_plane_size;
    for (int ky = 0; ky < kernel_h; ky++) {
      int weight_y = ky * kernel_w;
      for (int kx = 0; kx < kernel_w; kx++) {
        int weight_offset = weight_z + weight_y + kx;
        const float* weight_addr = weights + weight_offset;
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
          float* src_addr_h0 = boundary_y0 ? (src + src_z + src_y) : zero_ptr;
          float* src_addr_h1 =
              boundary_y1 ? (src + src_z + width * 2 + src_y) : zero_ptr;
          float* src_addr_h2 =
              boundary_y2 ? (src + src_z + width * 4 + src_y) : zero_ptr;
          float* src_addr_h3 =
              boundary_y3 ? (src + src_z + width * 6 + src_y) : zero_ptr;
          int iw = -pad_w0 + kx * dilation_w;
          int i = 0;

#ifdef __AVX2__  // _mm256_permute4x64_epi64 need avx2
          for (; i + 7 < output_w; i += 8, iw += 16) {
            int dst_offset = dst_z + dst_y + i;
            const float* dst_addr = dst + dst_offset;
            const float iw_data[8] = {iw + 0.f,
                                      iw + 2.f,
                                      iw + 4.f,
                                      iw + 6.f,
                                      iw + 8.f,
                                      iw + 10.f,
                                      iw + 12.f,
                                      iw + 14.f};

            // select weight
            __m256 vec_iw = _mm256_loadu_ps(&iw_data[0]);
            __m256 vec_mask = _mm256_and_ps(
                _mm256_cmp_ps(vec_iw, vec_zero, 13),
                _mm256_cmp_ps(vec_iw, vec_width, 1));  // GE:13  LT:1
            __m256 vec_weight = _mm256_set1_ps(weight_addr[0]);
            vec_weight = _mm256_blendv_ps(vec_zero, vec_weight, vec_mask);

            // compute 4 lines
            __m256 vec_data_lo = _mm256_loadu_ps(src_addr_h0 + iw);
            __m256 vec_data_hi = _mm256_loadu_ps(src_addr_h0 + iw + 8);
            __m256 vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            __m256i vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);
            __m256 vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr), vec_weight, vec_data);
            __m256 vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            __m256 vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h0 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h0 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));

            vec_data_lo = _mm256_loadu_ps(src_addr_h1 + iw);
            vec_data_hi = _mm256_loadu_ps(src_addr_h1 + iw + 8);
            vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);

            vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr + output_w), vec_weight, vec_data);
            vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h1 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h1 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));

            vec_data_lo = _mm256_loadu_ps(src_addr_h2 + iw);
            vec_data_hi = _mm256_loadu_ps(src_addr_h2 + iw + 8);
            vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);
            vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr + 2 * output_w), vec_weight, vec_data);
            vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h2 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h2 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));

            vec_data_lo = _mm256_loadu_ps(src_addr_h3 + iw);
            vec_data_hi = _mm256_loadu_ps(src_addr_h3 + iw + 8);
            vec_data =
                _mm256_shuffle_ps(vec_data_lo, vec_data_hi, 136);  // 0x88
            vec_tmp_data =
                _mm256_permute4x64_epi64(_mm256_castps_si256(vec_data),
                                         216);  // 11011000b
            vec_data = _mm256_castsi256_ps(vec_tmp_data);
            vec_dst = _mm256_fmadd_ps(
                _mm256_loadu_ps(dst_addr + 3 * output_w), vec_weight, vec_data);
            vec_dst_lo = _mm256_unpacklo_ps(vec_dst, vec_zero);
            vec_dst_hi = _mm256_unpackhi_ps(vec_dst, vec_zero);
            _mm256_maskstore_ps(
                src_addr_h3 + iw,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x20));
            _mm256_maskstore_ps(
                src_addr_h3 + iw + 8,
                vec_store_mask,
                _mm256_permute2f128_ps(vec_dst_lo, vec_dst_hi, 0x31));
          }
#endif
#ifdef __SSE4_1__
          for (; i + 3 < output_w; i += 4, iw += 8) {
            int dst_offset = dst_z + dst_y + i;
            const float* dst_addr = dst + dst_offset;
            const float iw_data[4] = {iw + 0.f, iw + 2.f, iw + 4.f, iw + 6.f};

            // select weight
            __m128 vec_iw_128 = _mm_loadu_ps(&iw_data[0]);
            __m128 vec_mask_128 =
                _mm_and_ps(_mm_cmpge_ps(vec_iw_128, vec_zero_128),
                           _mm_cmplt_ps(vec_iw_128, vec_width_128));
            __m128 vec_weight_128 = _mm_set1_ps(weight_addr[0]);
            vec_weight_128 =
                _mm_blendv_ps(vec_zero_128, vec_weight_128, vec_mask_128);

            // compute 4 lines
            __m128 vec_data_lo128 = _mm_loadu_ps(src_addr_h0 + iw);
            __m128 vec_data_hi128 = _mm_loadu_ps(src_addr_h0 + iw + 4);
            __m128 vec_data_128 =
                _mm_shuffle_ps(vec_data_lo128, vec_data_hi128, 136);  // 0x88
            __m128 vec_dst_128 =
                _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(dst_addr), vec_weight_128),
                           vec_data_128);
            _mm_storeu_ps(
                src_addr_h0 + iw,
                _mm_blend_ps(vec_data_lo128,
                             _mm_unpacklo_ps(vec_dst_128, vec_zero_128),
                             5));
            _mm_storeu_ps(
                src_addr_h0 + iw + 4,
                _mm_blend_ps(vec_data_hi128,
                             _mm_unpackhi_ps(vec_dst_128, vec_zero_128),
                             5));

            vec_data_lo128 = _mm_loadu_ps(src_addr_h1 + iw);
            vec_data_hi128 = _mm_loadu_ps(src_addr_h1 + iw + 4);
            vec_data_128 =
                _mm_shuffle_ps(vec_data_lo128, vec_data_hi128, 136);  // 0x88
            vec_dst_128 = _mm_add_ps(
                _mm_mul_ps(_mm_loadu_ps(dst_addr + output_w), vec_weight_128),
                vec_data_128);
            _mm_storeu_ps(
                src_addr_h1 + iw,
                _mm_blend_ps(vec_data_lo128,
                             _mm_unpacklo_ps(vec_dst_128, vec_zero_128),
                             5));
            _mm_storeu_ps(
                src_addr_h1 + iw + 4,
                _mm_blend_ps(vec_data_hi128,
                             _mm_unpackhi_ps(vec_dst_128, vec_zero_128),
                             5));

            vec_data_lo128 = _mm_loadu_ps(src_addr_h2 + iw);
            vec_data_hi128 = _mm_loadu_ps(src_addr_h2 + iw + 4);
            vec_data_128 =
                _mm_shuffle_ps(vec_data_lo128, vec_data_hi128, 136);  // 0x88
            vec_dst_128 =
                _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(dst_addr + 2 * output_w),
                                      vec_weight_128),
                           vec_data_128);
            _mm_storeu_ps(
                src_addr_h2 + iw,
                _mm_blend_ps(vec_data_lo128,
                             _mm_unpacklo_ps(vec_dst_128, vec_zero_128),
                             5));
            _mm_storeu_ps(
                src_addr_h2 + iw + 4,
                _mm_blend_ps(vec_data_hi128,
                             _mm_unpackhi_ps(vec_dst_128, vec_zero_128),
                             5));

            vec_data_lo128 = _mm_loadu_ps(src_addr_h3 + iw);
            vec_data_hi128 = _mm_loadu_ps(src_addr_h3 + iw + 4);
            vec_data_128 =
                _mm_shuffle_ps(vec_data_lo128, vec_data_hi128, 136);  // 0x88
            vec_dst_128 =
                _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(dst_addr + 3 * output_w),
                                      vec_weight_128),
                           vec_data_128);
            _mm_storeu_ps(
                src_addr_h3 + iw,
                _mm_blend_ps(vec_data_lo128,
                             _mm_unpacklo_ps(vec_dst_128, vec_zero_128),
                             5));
            _mm_storeu_ps(
                src_addr_h3 + iw + 4,
                _mm_blend_ps(vec_data_hi128,
                             _mm_unpackhi_ps(vec_dst_128, vec_zero_128),
                             5));
          }
#endif
          for (; i < output_w; i++, iw += 2) {
            bool boundary_x = ((iw >= 0) && (iw < width));
            int src_offset = src_z + src_y + iw;
            int dst_offset = dst_z + dst_y + i;
            src[src_offset] += (boundary_x) * (boundary_y0)*dst[dst_offset] *
                               weights[weight_offset];
            src[src_offset + width * 2] +=
                (boundary_x) * (boundary_y1)*dst[dst_offset + output_w] *
                weights[weight_offset];
            src[src_offset + width * 4] +=
                (boundary_x) * (boundary_y2)*dst[dst_offset + output_w * 2] *
                weights[weight_offset];
            src[src_offset + width * 6] +=
                (boundary_x) * (boundary_y3)*dst[dst_offset + output_w * 3] *
                weights[weight_offset];
          }
        }
      }
    }
  }
  TargetFree(TARGET(kX86), zero_ptr);
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
