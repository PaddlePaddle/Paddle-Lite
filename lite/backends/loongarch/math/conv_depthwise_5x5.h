/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include "lite/backends/loongarch/xxl.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

#ifdef __loongarch_asx
#define loadu_ps(a) lasx_loadu_f32(a)
#define fmadd_ps(a, b, c) lasx_fmadd_f32(a, b, c)
#define storeu_ps(a, b) lasx_storeu_f32(a, b)
#define setzero_ps() lasx_setzero_f32()
#define max_ps(a, b) lasx_max_f32(a, b)
#define min_ps(a, b) lasx_min_f32(a, b)
#define set1_ps(a) lasx_set1_f32(a)
#define mul_ps(a, b) lasx_mul_f32(a, b)
#define cmp_ps(a, b, c) lasx_cmp_f32(a, b, c)
#define blendv_ps(a, b, c) lasx_blendv_f32(a, b, c)
#define add_ps(a, b) lasx_add_f32(a, b)
#define block_channel 8
#define Type __m256
#else
#define loadu_ps(a) lsx_loadu_f32(a)
#define storeu_ps(a, b) lsx_storeu_f32(a, b)
#define fmadd_ps(a, b, c) lsx_add_f32(lsx_mul_f32(a, b), c)
#define setzero_ps() lsx_setzero_f32()
#define max_ps(a, b) lsx_max_f32(a, b)
#define min_ps(a, b) lsx_min_f32(a, b)
#define set1_ps(a) lsx_set1_f32(a)
#define mul_ps(a, b) lsx_mul_f32(a, b)
#define cmp_ps(a, b, c) lsx_cmp_f32(a, b, c)
#define blendv_ps(a, b, c) lsx_blendv_f32(a, b, c)
#define add_ps(a, b) lsx_add_f32(a, b)
#define block_channel 4
#define Type __m128
#endif

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
