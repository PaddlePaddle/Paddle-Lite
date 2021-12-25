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

#ifdef __AVX__
#include <immintrin.h>
#else
#include <emmintrin.h>
#include <smmintrin.h>
#endif

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

#ifdef __AVX__
#define loadu_ps(a) _mm256_loadu_ps(a)
#define fmadd_ps(a, b, c) _mm256_fmadd_ps(a, b, c)
#define storeu_ps(a, b) _mm256_storeu_ps(a, b)
#define setzero_ps() _mm256_setzero_ps()
#define max_ps(a, b) _mm256_max_ps(a, b)
#define min_ps(a, b) _mm256_min_ps(a, b)
#define set1_ps(a) _mm256_set1_ps(a)
#define mul_ps(a, b) _mm256_mul_ps(a, b)
#define cmp_ps(a, b, c) _mm256_cmp_ps(a, b, c)
#define blendv_ps(a, b, c) _mm256_blendv_ps(a, b, c)
#define add_ps(a, b) _mm256_add_ps(a, b)
#define block_channel 8
#define Type __m256
#else
#define loadu_ps(a) _mm_loadu_ps(a)
#define storeu_ps(a, b) _mm_storeu_ps(a, b)
#define fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define setzero_ps() _mm_setzero_ps()
#define max_ps(a, b) _mm_max_ps(a, b)
#define min_ps(a, b) _mm_min_ps(a, b)
#define set1_ps(a) _mm_set1_ps(a)
#define mul_ps(a, b) _mm_mul_ps(a, b)
#define cmp_ps(a, b, c) _mm_cmp_ps(a, b, c)
#define blendv_ps(a, b, c) _mm_blendv_ps(a, b, c)
#define add_ps(a, b) _mm_add_ps(a, b)
#define block_channel 4
#define Type __m128
#endif

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
