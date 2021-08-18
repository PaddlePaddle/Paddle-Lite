/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "lite/backends/x86/math/conv_direct.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "lite/backends/x86/math/conv_utils.h"
#include "lite/core/context.h"
#ifdef __AVX__
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void conv_direct_3x3s2(const float* i_data,
                       const float* trans_weight,
                       int bs,
                       int ic,
                       int ih,
                       int iw,
                       int oc,
                       int oc_expand,
                       float* o_data,
                       int oh,
                       int ow,
                       int ph,
                       int pw,
                       const float* bias,
                       lite_api::ActivationType active_type) {
  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int strideh = 2;
  constexpr int stridew = 2;

#ifdef __AVX__
  constexpr int BLOCK = 8;
  // the sliding window is 5x7 and can obtain 2x3 results！ for AVX
  constexpr int window_h = 5;
  constexpr int window_w = 7;

#else
  constexpr int BLOCK = 4;
  constexpr int window_h = 5;
  constexpr int window_w = 7;
#endif

  // The maximum value of the upper left corner of the
  // sliding window in h dimension
  int new_ih;
  int new_iw;
  int new_ih_start;
  int new_iw_start;
  if (ph == 0 && pw == 0) {
    // 4 is the stride_h of sliding window
    // 6 is the stride_w of sliding window
    new_ih = (ih - window_h) / 4 * 4;
    new_iw = (iw - window_w) / 6 * 6;
    new_ih_start = 0;
    new_iw_start = 0;
  } else if (ph == 1 && pw == 1) {
    new_iw = (iw - window_w - 1) / 6 * 6 + 1;
    new_ih = (ih - window_h - 1) / 4 * 4 + 1;
    new_ih_start = 1;
    new_iw_start = 1;
  } else {
    LOG(FATAL) << "[X86] conv_direct only support 3x3s2 with padding = 0 or 1";
    ;
  }

  // [0,o_left) in output map needs Special treatment (Left boundary)
  // [o_right, ow) in output map needs Special treatment (Right boundary)
  // [0,o_upper) same as above (Upper boundary)
  // [o_down, oh) same as above (Lower boundary)
  int o_left = (new_iw_start + pw) / 2;
  int o_right = (new_iw + pw) / 2 + 3;
  int o_upper = (new_ih_start + ph) / 2;
  int o_down = (new_ih + ph) / 2 + 2;

  // The number of channels of convolution kernel
  // and the number of input channels are always the same !
  int wc = ic;

  //
  int ichw = ic * ih * iw;
  int ihw = ih * iw;
  int wchwB = wc * wh * ww * BLOCK;
  int whwB = wh * ww * BLOCK;
  int ohw = oh * ow;
  int ochw = oc * oh * ow;
  int ohwB = oh * ow * BLOCK;
  int owB = ow * BLOCK;
  int trans_out_size = oc_expand / BLOCK * ohwB;

  // holds the intermediate  HWC output result
  float* trans_out = static_cast<float*>(
      TargetMalloc(TARGET(kX86), sizeof(float) * trans_out_size));

  // fetch bs_i th input feature map
  for (int bs_i = 0; bs_i < bs; bs_i++) {
    memset(trans_out, 0, sizeof(float) * trans_out_size);
    // fetch the ic_i th channel in this input feature map
    for (int ic_i = 0; ic_i < wc; ic_i++) {
      const float* input_start_address = i_data + bs_i * ichw + ic_i * ihw;

      // fetch oc_gi th group kernel,there are BLOCK kernels
      // in it. we only need to deal with its ic_i channel !
      // oc_gi is oc_group_i !
      for (int oc_gi = 0; oc_gi < oc_expand / BLOCK; oc_gi++) {
        // Now, we need compute the conv of one planar feature map and BLOCK
        // planar kernel
        // the  planar feature map's starting address
        const float* kernel_start_address =
            trans_weight + oc_gi * wchwB +
            ic_i * whwB;  // the first kernel's address in this BLOCK
        float* output_start_address = trans_out + oc_gi * ohwB;

        // Handle upper boundary！
        for (int oh_i = 0; oh_i < o_upper; oh_i++) {
          for (int ow_i = 0; ow_i < ow; ow_i++) {
            float* output_address =
                output_start_address + oh_i * ow * BLOCK + ow_i * BLOCK;

            // oh_i and ow_i is the index of the output.
            // Next, calculate the index of their corresponding input.
            // These two are in the upper left corner of the corresponding
            // input!
            int ih_i = oh_i * strideh - ph;
            int iw_i = ow_i * stridew - pw;

// Let's start the convolution of 3x3!
#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
        // Handle lower boundary！
        for (int oh_i = o_down; oh_i < oh; oh_i++) {
          for (int ow_i = 0; ow_i < ow; ow_i++) {
            float* output_address =
                output_start_address + oh_i * ow * BLOCK + ow_i * BLOCK;
            int ih_i = oh_i * strideh - ph;
            int iw_i = ow_i * stridew - pw;
#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }

        // Handle left boundary！
        for (int oh_i = 0; oh_i < oh; oh_i++) {
          if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
            continue;
          for (int ow_i = 0; ow_i < o_left; ow_i++) {
            float* output_address =
                output_start_address + oh_i * ow * BLOCK + ow_i * BLOCK;
            int ih_i = oh_i * strideh - ph;
            int iw_i = ow_i * stridew - pw;
#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
        // Handle right boundary！
        for (int oh_i = 0; oh_i < oh; oh_i++) {
          if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
            continue;
          for (int ow_i = o_right; ow_i < ow; ow_i++) {
            float* output_address =
                output_start_address + oh_i * ow * BLOCK + ow_i * BLOCK;
            int ih_i = oh_i * strideh - ph;
            int iw_i = ow_i * stridew - pw;
#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }

// prefetch the 3x3 conv kernel outside the below two Nested loop !
#ifdef __AVX__

        // Take out 9 weight values to the register
        __m256 w00 = _mm256_loadu_ps(kernel_start_address + 0 * BLOCK);
        __m256 w01 = _mm256_loadu_ps(kernel_start_address + 1 * BLOCK);
        __m256 w02 = _mm256_loadu_ps(kernel_start_address + 2 * BLOCK);
        __m256 w10 = _mm256_loadu_ps(kernel_start_address + 3 * BLOCK);
        __m256 w11 = _mm256_loadu_ps(kernel_start_address + 4 * BLOCK);
        __m256 w12 = _mm256_loadu_ps(kernel_start_address + 5 * BLOCK);
        __m256 w20 = _mm256_loadu_ps(kernel_start_address + 6 * BLOCK);
        __m256 w21 = _mm256_loadu_ps(kernel_start_address + 7 * BLOCK);
        __m256 w22 = _mm256_loadu_ps(kernel_start_address + 8 * BLOCK);
#else
        // Take out 9 weight values to the register
        __m128 w00 = _mm_loadu_ps(kernel_start_address + 0 * BLOCK);
        __m128 w01 = _mm_loadu_ps(kernel_start_address + 1 * BLOCK);
        __m128 w02 = _mm_loadu_ps(kernel_start_address + 2 * BLOCK);
        __m128 w10 = _mm_loadu_ps(kernel_start_address + 3 * BLOCK);
        __m128 w11 = _mm_loadu_ps(kernel_start_address + 4 * BLOCK);
        __m128 w12 = _mm_loadu_ps(kernel_start_address + 5 * BLOCK);
        __m128 w20 = _mm_loadu_ps(kernel_start_address + 6 * BLOCK);
        __m128 w21 = _mm_loadu_ps(kernel_start_address + 7 * BLOCK);
        __m128 w22 = _mm_loadu_ps(kernel_start_address + 8 * BLOCK);

#endif

        // So far, we have dealt with the special boundary,
        // and now we begin to deal with the general situation
        for (int ih_i = new_ih_start; ih_i <= new_ih; ih_i += 4) {
          // one sliding window cangenerate 2x3 results
          // below is the two line's first address!
          float* output_address0 = output_start_address +
                                   (ih_i + ph) / 2 * ow * BLOCK +
                                   (new_iw_start + pw) / 2 * BLOCK;
          float* output_address1 = output_address0 + ow * BLOCK;

          // iv is (ih_i~ ih_i + 4)row's first address !
          const float* row0 = input_start_address + ih_i * iw;
          const float* row1 = row0 + 1 * iw;
          const float* row2 = row0 + 2 * iw;
          const float* row3 = row0 + 3 * iw;
          const float* row4 = row0 + 4 * iw;

          for (int iw_i = new_iw_start; iw_i <= new_iw; iw_i += 6) {
            // The following is the starting address of
            // each line of the sliding window
            const float* iv0 = row0 + iw_i;
            const float* iv1 = row1 + iw_i;
            const float* iv2 = row2 + iw_i;
            const float* iv3 = row3 + iw_i;
            const float* iv4 = row4 + iw_i;
            // the first line output's address
            float* output_address00 = output_address0 + BLOCK * 0;
            float* output_address01 = output_address0 + BLOCK * 1;
            float* output_address02 = output_address0 + BLOCK * 2;
            // the second line output's address
            float* output_address10 = output_address1 + BLOCK * 0;
            float* output_address11 = output_address1 + BLOCK * 1;
            float* output_address12 = output_address1 + BLOCK * 2;

#ifdef __AVX__

            // Sliding windows can produce 2x3 results, but I now create three
            // __m256
            // to  represent the outputs  in the first line
            __m256 res00 = _mm256_loadu_ps(output_address00);
            __m256 res01 = _mm256_loadu_ps(output_address01);
            __m256 res02 = _mm256_loadu_ps(output_address02);

            // I have used 12 registers, and there are 4 left!
            // Next, I will use three to hold input data to generate outputs !
            // iv0: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            __m256 input0 = _mm256_set1_ps(iv0[0]);
            __m256 input2 = _mm256_set1_ps(
                iv0[2]);  // needs to be retained because of reuse
            __m256 input4 = _mm256_set1_ps(
                iv0[4]);  // This needs to be retained because of reuse
            res00 = _mm256_fmadd_ps(input0, w00, res00);
            res01 = _mm256_fmadd_ps(input2, w00, res01);
            res02 = _mm256_fmadd_ps(input4, w00, res02);
            input0 = _mm256_set1_ps(iv0[6]);
            res00 = _mm256_fmadd_ps(input2, w02, res00);
            res01 = _mm256_fmadd_ps(input4, w02, res01);
            res02 = _mm256_fmadd_ps(input0, w02, res02);
            input0 = _mm256_set1_ps(iv0[1]);
            input2 = _mm256_set1_ps(iv0[3]);
            input4 = _mm256_set1_ps(iv0[5]);
            res00 = _mm256_fmadd_ps(input0, w01, res00);
            res01 = _mm256_fmadd_ps(input2, w01, res01);
            res02 = _mm256_fmadd_ps(input4, w01, res02);
            // iv1: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm256_set1_ps(iv1[0]);
            input2 = _mm256_set1_ps(
                iv1[2]);  // This needs to be retained because of reuse
            input4 = _mm256_set1_ps(
                iv1[4]);  // This needs to be retained because of reuse
            res00 = _mm256_fmadd_ps(input0, w10, res00);
            res01 = _mm256_fmadd_ps(input2, w10, res01);
            res02 = _mm256_fmadd_ps(input4, w10, res02);
            input0 = _mm256_set1_ps(iv1[6]);
            res00 = _mm256_fmadd_ps(input2, w12, res00);
            res01 = _mm256_fmadd_ps(input4, w12, res01);
            res02 = _mm256_fmadd_ps(input0, w12, res02);
            input0 = _mm256_set1_ps(iv1[1]);
            input2 = _mm256_set1_ps(iv1[3]);
            input4 = _mm256_set1_ps(iv1[5]);
            res00 = _mm256_fmadd_ps(input0, w11, res00);
            res01 = _mm256_fmadd_ps(input2, w11, res01);
            res02 = _mm256_fmadd_ps(input4, w11, res02);
            // iv2: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm256_set1_ps(iv2[0]);
            input2 = _mm256_set1_ps(iv2[2]);
            input4 = _mm256_set1_ps(iv2[4]);
            res00 = _mm256_fmadd_ps(input0, w20, res00);
            res01 = _mm256_fmadd_ps(input2, w20, res01);
            res02 = _mm256_fmadd_ps(input4, w20, res02);
            input0 = _mm256_set1_ps(iv2[6]);
            res00 = _mm256_fmadd_ps(input2, w22, res00);
            res01 = _mm256_fmadd_ps(input4, w22, res01);
            res02 = _mm256_fmadd_ps(input0, w22, res02);
            input0 = _mm256_set1_ps(iv2[1]);
            input2 = _mm256_set1_ps(iv2[3]);
            input4 = _mm256_set1_ps(iv2[5]);
            res00 = _mm256_fmadd_ps(input0, w21, res00);
            res01 = _mm256_fmadd_ps(input2, w21, res01);
            res02 = _mm256_fmadd_ps(input4, w21, res02);
            // The first, second and third results have been calculated above.
            // Store them back
            _mm256_storeu_ps(output_address00, res00);
            _mm256_storeu_ps(output_address01, res01);
            _mm256_storeu_ps(output_address02, res02);

            // res00,res01,res02 need to be updated representing the outputs in
            // the second line
            // iv2: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            res00 = _mm256_loadu_ps(output_address10);
            res01 = _mm256_loadu_ps(output_address11);
            res02 = _mm256_loadu_ps(output_address12);
            res00 = _mm256_fmadd_ps(input0, w01, res00);
            res01 = _mm256_fmadd_ps(input2, w01, res01);
            res02 = _mm256_fmadd_ps(input4, w01, res02);
            input0 = _mm256_set1_ps(iv2[0]);
            input2 = _mm256_set1_ps(iv2[2]);
            input4 = _mm256_set1_ps(iv2[4]);
            res00 = _mm256_fmadd_ps(input0, w00, res00);
            res01 = _mm256_fmadd_ps(input2, w00, res01);
            res02 = _mm256_fmadd_ps(input4, w00, res02);
            input0 = _mm256_set1_ps(iv2[6]);
            res00 = _mm256_fmadd_ps(input2, w02, res00);
            res01 = _mm256_fmadd_ps(input4, w02, res01);
            res02 = _mm256_fmadd_ps(input0, w02, res02);
            // iv3: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm256_set1_ps(iv3[0]);
            input2 = _mm256_set1_ps(iv3[2]);
            input4 = _mm256_set1_ps(iv3[4]);
            res00 = _mm256_fmadd_ps(input0, w10, res00);
            res01 = _mm256_fmadd_ps(input2, w10, res01);
            res02 = _mm256_fmadd_ps(input4, w10, res02);
            input0 = _mm256_set1_ps(iv3[6]);
            res00 = _mm256_fmadd_ps(input2, w12, res00);
            res01 = _mm256_fmadd_ps(input4, w12, res01);
            res02 = _mm256_fmadd_ps(input0, w12, res02);
            input0 = _mm256_set1_ps(iv3[1]);
            input2 = _mm256_set1_ps(iv3[3]);
            input4 = _mm256_set1_ps(iv3[5]);
            res00 = _mm256_fmadd_ps(input0, w11, res00);
            res01 = _mm256_fmadd_ps(input2, w11, res01);
            res02 = _mm256_fmadd_ps(input4, w11, res02);
            // iv4: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm256_set1_ps(iv4[0]);
            input2 = _mm256_set1_ps(iv4[2]);
            input4 = _mm256_set1_ps(iv4[4]);
            res00 = _mm256_fmadd_ps(input0, w20, res00);
            res01 = _mm256_fmadd_ps(input2, w20, res01);
            res02 = _mm256_fmadd_ps(input4, w20, res02);
            input0 = _mm256_set1_ps(iv4[6]);
            res00 = _mm256_fmadd_ps(input2, w22, res00);
            res01 = _mm256_fmadd_ps(input4, w22, res01);
            res02 = _mm256_fmadd_ps(input0, w22, res02);
            input0 = _mm256_set1_ps(iv4[1]);
            input2 = _mm256_set1_ps(iv4[3]);
            input4 = _mm256_set1_ps(iv4[5]);
            res00 = _mm256_fmadd_ps(input0, w21, res00);
            res01 = _mm256_fmadd_ps(input2, w21, res01);
            res02 = _mm256_fmadd_ps(input4, w21, res02);
            // Store them back to trans_out!
            _mm256_storeu_ps(output_address10, res00);
            _mm256_storeu_ps(output_address11, res01);
            _mm256_storeu_ps(output_address12, res02);

#else
            // Sliding windows can produce 2x3 results, but I now create three
            // __m256
            // to  represent the outputs  in the first line
            __m128 res00 = _mm_loadu_ps(output_address00);
            __m128 res01 = _mm_loadu_ps(output_address01);
            __m128 res02 = _mm_loadu_ps(output_address02);

            // I have used 12 registers, and there are 4 left!
            // Next, I will use three to hold input data to generate outputs !
            // iv0: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            __m128 input0 = _mm_set1_ps(iv0[0]);
            __m128 input2 = _mm_set1_ps(
                iv0[2]);  // This needs to be retained because of reuse
            __m128 input4 = _mm_set1_ps(
                iv0[4]);  // This needs to be retained because of reuse
            res00 = _mm_fmadd_ps(input0, w00, res00);
            res01 = _mm_fmadd_ps(input2, w00, res01);
            res02 = _mm_fmadd_ps(input4, w00, res02);
            input0 = _mm_set1_ps(iv0[6]);
            res00 = _mm_fmadd_ps(input2, w02, res00);
            res01 = _mm_fmadd_ps(input4, w02, res01);
            res02 = _mm_fmadd_ps(input0, w02, res02);
            input0 = _mm_set1_ps(iv0[1]);
            input2 = _mm_set1_ps(iv0[3]);
            input4 = _mm_set1_ps(iv0[5]);
            res00 = _mm_fmadd_ps(input0, w01, res00);
            res01 = _mm_fmadd_ps(input2, w01, res01);
            res02 = _mm_fmadd_ps(input4, w01, res02);
            // iv1: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm_set1_ps(iv1[0]);
            input2 = _mm_set1_ps(
                iv1[2]);  // This needs to be retained because of reuse
            input4 = _mm_set1_ps(
                iv1[4]);  // This needs to be retained because of reuse
            res00 = _mm_fmadd_ps(input0, w10, res00);
            res01 = _mm_fmadd_ps(input2, w10, res01);
            res02 = _mm_fmadd_ps(input4, w10, res02);
            input0 = _mm_set1_ps(iv1[6]);
            res00 = _mm_fmadd_ps(input2, w12, res00);
            res01 = _mm_fmadd_ps(input4, w12, res01);
            res02 = _mm_fmadd_ps(input0, w12, res02);
            input0 = _mm_set1_ps(iv1[1]);
            input2 = _mm_set1_ps(iv1[3]);
            input4 = _mm_set1_ps(iv1[5]);
            res00 = _mm_fmadd_ps(input0, w11, res00);
            res01 = _mm_fmadd_ps(input2, w11, res01);
            res02 = _mm_fmadd_ps(input4, w11, res02);
            // iv2: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm_set1_ps(iv2[0]);
            input2 = _mm_set1_ps(iv2[2]);
            input4 = _mm_set1_ps(iv2[4]);
            res00 = _mm_fmadd_ps(input0, w20, res00);
            res01 = _mm_fmadd_ps(input2, w20, res01);
            res02 = _mm_fmadd_ps(input4, w20, res02);
            input0 = _mm_set1_ps(iv2[6]);
            res00 = _mm_fmadd_ps(input2, w22, res00);
            res01 = _mm_fmadd_ps(input4, w22, res01);
            res02 = _mm_fmadd_ps(input0, w22, res02);
            input0 = _mm_set1_ps(iv2[1]);
            input2 = _mm_set1_ps(iv2[3]);
            input4 = _mm_set1_ps(iv2[5]);
            res00 = _mm_fmadd_ps(input0, w21, res00);
            res01 = _mm_fmadd_ps(input2, w21, res01);
            res02 = _mm_fmadd_ps(input4, w21, res02);
            // The first, second and third results have been calculated above.
            // Store them back
            _mm_storeu_ps(output_address00, res00);
            _mm_storeu_ps(output_address01, res01);
            _mm_storeu_ps(output_address02, res02);

            // res00,res01,res02 need to be updated representing the outputs in
            // the second line
            // iv2: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            res00 = _mm_loadu_ps(output_address10);
            res01 = _mm_loadu_ps(output_address11);
            res02 = _mm_loadu_ps(output_address12);
            res00 = _mm_fmadd_ps(input0, w01, res00);
            res01 = _mm_fmadd_ps(input2, w01, res01);
            res02 = _mm_fmadd_ps(input4, w01, res02);
            input0 = _mm_set1_ps(iv2[0]);
            input2 = _mm_set1_ps(iv2[2]);
            input4 = _mm_set1_ps(iv2[4]);
            res00 = _mm_fmadd_ps(input0, w00, res00);
            res01 = _mm_fmadd_ps(input2, w00, res01);
            res02 = _mm_fmadd_ps(input4, w00, res02);
            input0 = _mm_set1_ps(iv2[6]);
            res00 = _mm_fmadd_ps(input2, w02, res00);
            res01 = _mm_fmadd_ps(input4, w02, res01);
            res02 = _mm_fmadd_ps(input0, w02, res02);
            // iv3: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm_set1_ps(iv3[0]);
            input2 = _mm_set1_ps(iv3[2]);
            input4 = _mm_set1_ps(iv3[4]);
            res00 = _mm_fmadd_ps(input0, w10, res00);
            res01 = _mm_fmadd_ps(input2, w10, res01);
            res02 = _mm_fmadd_ps(input4, w10, res02);
            input0 = _mm_set1_ps(iv3[6]);
            res00 = _mm_fmadd_ps(input2, w12, res00);
            res01 = _mm_fmadd_ps(input4, w12, res01);
            res02 = _mm_fmadd_ps(input0, w12, res02);
            input0 = _mm_set1_ps(iv3[1]);
            input2 = _mm_set1_ps(iv3[3]);
            input4 = _mm_set1_ps(iv3[5]);
            res00 = _mm_fmadd_ps(input0, w11, res00);
            res01 = _mm_fmadd_ps(input2, w11, res01);
            res02 = _mm_fmadd_ps(input4, w11, res02);
            // iv4: 0 1 2 3 4 5 6
            // 0,1,2 is Responsible for res00
            // 2,3,4 is Responsible for res01
            // 4,5,6 is Responsible for res01
            input0 = _mm_set1_ps(iv4[0]);
            input2 = _mm_set1_ps(iv4[2]);
            input4 = _mm_set1_ps(iv4[4]);
            res00 = _mm_fmadd_ps(input0, w20, res00);
            res01 = _mm_fmadd_ps(input2, w20, res01);
            res02 = _mm_fmadd_ps(input4, w20, res02);
            input0 = _mm_set1_ps(iv4[6]);
            res00 = _mm_fmadd_ps(input2, w22, res00);
            res01 = _mm_fmadd_ps(input4, w22, res01);
            res02 = _mm_fmadd_ps(input0, w22, res02);
            input0 = _mm_set1_ps(iv4[1]);
            input2 = _mm_set1_ps(iv4[3]);
            input4 = _mm_set1_ps(iv4[5]);
            res00 = _mm_fmadd_ps(input0, w21, res00);
            res01 = _mm_fmadd_ps(input2, w21, res01);
            res02 = _mm_fmadd_ps(input4, w21, res02);
            // Store them back to trans_out!
            _mm_storeu_ps(output_address10, res00);
            _mm_storeu_ps(output_address11, res01);
            _mm_storeu_ps(output_address12, res02);
#endif

            // when inner loop ends, the output_address needs to be increased by
            // 3 * BLOCK
            output_address0 += 3 * BLOCK;
            output_address1 += 3 * BLOCK;
          }
        }
      }
    }

    // convert trans_out(HWC) to o_data(CHW)!
    for (int oc_gi = 0; oc_gi < oc / BLOCK; oc_gi++) {
      for (int oh_i = 0; oh_i < oh; oh_i += 1) {
        for (int ow_i = 0; ow_i < ow / BLOCK * BLOCK; ow_i += BLOCK) {
          // trans_out's start_index, we need fetch 8x8 element;
          float* from_address =
              trans_out + oc_gi * ohwB + oh_i * owB + ow_i * BLOCK;

#ifdef __AVX__
          __m256 row0 = _mm256_loadu_ps(from_address + 0 * BLOCK);
          __m256 row1 = _mm256_loadu_ps(from_address + 1 * BLOCK);
          __m256 row2 = _mm256_loadu_ps(from_address + 2 * BLOCK);
          __m256 row3 = _mm256_loadu_ps(from_address + 3 * BLOCK);
          __m256 row4 = _mm256_loadu_ps(from_address + 4 * BLOCK);
          __m256 row5 = _mm256_loadu_ps(from_address + 5 * BLOCK);
          __m256 row6 = _mm256_loadu_ps(from_address + 6 * BLOCK);
          __m256 row7 = _mm256_loadu_ps(from_address + 7 * BLOCK);
          transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
#else

          __m128 row0 = _mm_loadu_ps(from_address + 0 * BLOCK);
          __m128 row1 = _mm_loadu_ps(from_address + 1 * BLOCK);
          __m128 row2 = _mm_loadu_ps(from_address + 2 * BLOCK);
          __m128 row3 = _mm_loadu_ps(from_address + 3 * BLOCK);
          _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
#endif

          if (bias != nullptr) {
#ifdef __AVX__
            row0 = _mm256_add_ps(row0, _mm256_set1_ps(bias[oc_gi * BLOCK + 0]));
            row1 = _mm256_add_ps(row1, _mm256_set1_ps(bias[oc_gi * BLOCK + 1]));
            row2 = _mm256_add_ps(row2, _mm256_set1_ps(bias[oc_gi * BLOCK + 2]));
            row3 = _mm256_add_ps(row3, _mm256_set1_ps(bias[oc_gi * BLOCK + 3]));
            row4 = _mm256_add_ps(row4, _mm256_set1_ps(bias[oc_gi * BLOCK + 4]));
            row5 = _mm256_add_ps(row5, _mm256_set1_ps(bias[oc_gi * BLOCK + 5]));
            row6 = _mm256_add_ps(row6, _mm256_set1_ps(bias[oc_gi * BLOCK + 6]));
            row7 = _mm256_add_ps(row7, _mm256_set1_ps(bias[oc_gi * BLOCK + 7]));
#else
            row0 = _mm_add_ps(row0, _mm_set1_ps(bias[oc_gi * BLOCK + 0]));
            row1 = _mm_add_ps(row1, _mm_set1_ps(bias[oc_gi * BLOCK + 1]));
            row2 = _mm_add_ps(row2, _mm_set1_ps(bias[oc_gi * BLOCK + 2]));
            row3 = _mm_add_ps(row3, _mm_set1_ps(bias[oc_gi * BLOCK + 3]));
#endif
          }
          if (active_type == lite_api::ActivationType::kRelu) {
#ifdef __AVX__
            row0 = _mm256_max_ps(row0, _mm256_set1_ps(0.f));
            row1 = _mm256_max_ps(row1, _mm256_set1_ps(0.f));
            row2 = _mm256_max_ps(row2, _mm256_set1_ps(0.f));
            row3 = _mm256_max_ps(row3, _mm256_set1_ps(0.f));
            row4 = _mm256_max_ps(row4, _mm256_set1_ps(0.f));
            row5 = _mm256_max_ps(row5, _mm256_set1_ps(0.f));
            row6 = _mm256_max_ps(row6, _mm256_set1_ps(0.f));
            row7 = _mm256_max_ps(row7, _mm256_set1_ps(0.f));
#else
            row0 = _mm_max_ps(row0, _mm_set1_ps(0.f));
            row1 = _mm_max_ps(row1, _mm_set1_ps(0.f));
            row2 = _mm_max_ps(row2, _mm_set1_ps(0.f));
            row3 = _mm_max_ps(row3, _mm_set1_ps(0.f));
#endif
          } else if (active_type == lite_api::ActivationType::kRelu6) {
#ifdef __AVX__
            row0 = _mm256_max_ps(row0, _mm256_set1_ps(0.f));
            row1 = _mm256_max_ps(row1, _mm256_set1_ps(0.f));
            row2 = _mm256_max_ps(row2, _mm256_set1_ps(0.f));
            row3 = _mm256_max_ps(row3, _mm256_set1_ps(0.f));
            row4 = _mm256_max_ps(row4, _mm256_set1_ps(0.f));
            row5 = _mm256_max_ps(row5, _mm256_set1_ps(0.f));
            row6 = _mm256_max_ps(row6, _mm256_set1_ps(0.f));
            row7 = _mm256_max_ps(row7, _mm256_set1_ps(0.f));
            row0 = _mm256_min_ps(row0, _mm256_set1_ps(6.f));
            row1 = _mm256_min_ps(row1, _mm256_set1_ps(6.f));
            row2 = _mm256_min_ps(row2, _mm256_set1_ps(6.f));
            row3 = _mm256_min_ps(row3, _mm256_set1_ps(6.f));
            row4 = _mm256_min_ps(row4, _mm256_set1_ps(6.f));
            row5 = _mm256_min_ps(row5, _mm256_set1_ps(6.f));
            row6 = _mm256_min_ps(row6, _mm256_set1_ps(6.f));
            row7 = _mm256_min_ps(row7, _mm256_set1_ps(6.f));

#else

            row0 = _mm_max_ps(row0, _mm_set1_ps(0.f));
            row1 = _mm_max_ps(row1, _mm_set1_ps(0.f));
            row2 = _mm_max_ps(row2, _mm_set1_ps(0.f));
            row3 = _mm_max_ps(row3, _mm_set1_ps(0.f));
            row0 = _mm_min_ps(row0, _mm_set1_ps(6.f));
            row1 = _mm_min_ps(row1, _mm_set1_ps(6.f));
            row2 = _mm_min_ps(row2, _mm_set1_ps(6.f));
            row3 = _mm_min_ps(row3, _mm_set1_ps(6.f));
#endif
          } else if (active_type == lite_api::ActivationType::kIndentity) {
          } else {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }

          float* dst_address =
              o_data + bs_i * ochw + oc_gi * ohwB + oh_i * ow + ow_i;
#ifdef __AVX__
          _mm256_storeu_ps(dst_address + 0 * ohw, row0);
          _mm256_storeu_ps(dst_address + 1 * ohw, row1);
          _mm256_storeu_ps(dst_address + 2 * ohw, row2);
          _mm256_storeu_ps(dst_address + 3 * ohw, row3);
          _mm256_storeu_ps(dst_address + 4 * ohw, row4);
          _mm256_storeu_ps(dst_address + 5 * ohw, row5);
          _mm256_storeu_ps(dst_address + 6 * ohw, row6);
          _mm256_storeu_ps(dst_address + 7 * ohw, row7);
#else
          _mm_storeu_ps(dst_address + 0 * ohw, row0);
          _mm_storeu_ps(dst_address + 1 * ohw, row1);
          _mm_storeu_ps(dst_address + 2 * ohw, row2);
          _mm_storeu_ps(dst_address + 3 * ohw, row3);
#endif
        }
        for (int ow_i = ow / BLOCK * BLOCK; ow_i < ow; ow_i++) {
          // trans_out
          float* from_address =
              trans_out + oc_gi * ohwB + oh_i * owB + ow_i * BLOCK;
          float* dst_address =
              o_data + bs_i * ochw + oc_gi * ohwB + oh_i * ow + ow_i;
#ifdef __AVX__
          __m256 row = _mm256_loadu_ps(from_address);
#else
          __m128 row = _mm_loadu_ps(from_address);
#endif
          if (bias != nullptr) {
#ifdef __AVX__
            row = _mm256_add_ps(row, _mm256_loadu_ps(&bias[oc_gi * BLOCK]));
#else
            row = _mm_add_ps(row, _mm_loadu_ps(&bias[oc_gi * BLOCK]));
#endif
          }
          if (active_type == lite_api::ActivationType::kRelu) {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
#endif
          } else if (active_type == lite_api::ActivationType::kRelu6) {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
            row = _mm256_min_ps(row, _mm256_set1_ps(6.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
            row = _mm_min_ps(row, _mm_set1_ps(6.f));
#endif
          } else if (active_type == lite_api::ActivationType::kIndentity) {
          } else {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }
#ifdef __AVX__
          *(dst_address + 0 * oh * ow) = ((float*)(&row))[0];
          *(dst_address + 1 * oh * ow) = ((float*)(&row))[1];
          *(dst_address + 2 * oh * ow) = ((float*)(&row))[2];
          *(dst_address + 3 * oh * ow) = ((float*)(&row))[3];
          *(dst_address + 4 * oh * ow) = ((float*)(&row))[4];
          *(dst_address + 5 * oh * ow) = ((float*)(&row))[5];
          *(dst_address + 6 * oh * ow) = ((float*)(&row))[6];
          *(dst_address + 7 * oh * ow) = ((float*)(&row))[7];
#else
          *(dst_address + 0 * oh * ow) = ((float*)(&row))[0];
          *(dst_address + 1 * oh * ow) = ((float*)(&row))[1];
          *(dst_address + 2 * oh * ow) = ((float*)(&row))[2];
          *(dst_address + 3 * oh * ow) = ((float*)(&row))[3];
#endif
        }
      }
    }
  }
  TargetFree(TARGET(kX86), trans_out);
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
