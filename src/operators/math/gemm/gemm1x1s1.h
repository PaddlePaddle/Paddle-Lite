/*
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef CONV_OP

#pragma once
#include "framework/tensor.h"

namespace paddle_mobile {
namespace operators {
namespace math {

#ifdef __aarch64__
const int MBLOCK = 8;
const int NBLOCK = 12;
const int KBLOCK = 4;
inline int get_hblock(ARMArch arch) { return MBLOCK; }
#else
const int MBLOCK_A73 = 4;
const int MBLOCK_OTH = 6;
const int NBLOCK = 8;
const int KBLOCK = 4;

inline int get_hblock(ARMArch arch) {
  if (arch == A73) {
    return MBLOCK_A73;
  } else {
    return MBLOCK_OTH;
  }
}
#endif  // __aarch64__

void gemm1x1s1_transform_weight(const framework::Tensor& weight,
                                const framework::Tensor& output,
                                framework::Tensor* trans_weight,
                                const int group, ARMArch arch);

void sgemm_prepack(const float* A_packed, const float* B, const float* bias,
                   float* C, int M, int N, int K, bool is_bias, bool is_relu,
                   bool is_transB, ARMArch arch);

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // CONV_OP
