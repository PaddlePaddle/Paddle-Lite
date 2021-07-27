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

#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE_4_2__)
#include <xmmintrin.h>
#endif


namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void fill_bias_fc(
    float *out, const float *bias, int num, int channel, bool flag_relu) {
#ifdef __AVX__
  __m256 vec_zero = _mm256_set1_ps(0.f);
  __m256 vec_bias = {0.f};
  __m256 vec_data = {0.f};
#endif
#ifdef __SSE4_2__
  __m128 vec_zero_128 = _mm_set1_ps(0.f);
  __m128 vec_bias_128 = {0.f};
  __m128 vec_data_128 = {0.f};
#endif
  int i = 0;

  if(flag_relu){
    for(int j = 0; j < num; j++){
      float *ptr = out + j * channel;
      const float *pbias = bias;
      i = 0;
#ifdef __AVX__
      for(; i + 7 < channel; i += 8)
      {
        vec_bias = _mm256_loadu_ps(pbias + i);
        vec_data = _mm256_loadu_ps(ptr + i);
        vec_data = _mm256_max_ps(_mm256_add_ps(vec_bias, vec_data), vec_zero);
        _mm256_storeu_ps(ptr + i, vec_data);
      }
      _mm256_zeroupper();
#endif
#ifdef __SSE4_2__
      for(; i + 3 < channel; i += 4)
      {
        vec_bias_128 = _mm_loadu_ps(pbias + i);
        vec_data_128 = _mm_loadu_ps(ptr + i);
        vec_data_128 = _mm_max_ps(_mm_add_ps(vec_bias_128, vec_data_128), vec_zero_128);
        _mm_storeu_ps(ptr + i, vec_data_128);
      }
#endif
      for(; i < channel; i++)
      {
        float tmp = pbias[i] + ptr[i];
        *(ptr + i) = tmp > 0.f ? tmp : 0.f;                 
      }
    }
  } else {
    for(int j = 0; j < num; j++){
      float *ptr = out + j * channel;
      const float *pbias = bias;
      i = 0;
#ifdef __AVX__
      for(; i + 7 < channel; i += 8)
      {
        vec_bias = _mm256_loadu_ps(pbias + i);
        vec_data = _mm256_loadu_ps(ptr + i);
        _mm256_storeu_ps(ptr + i, _mm256_add_ps(vec_data, vec_bias));
      }
      _mm256_zeroupper();
#endif
#ifdef __SSE4_2__
      for(; i + 3 < channel; i += 4)
      {
        vec_bias_128 = _mm_loadu_ps(pbias + i);
        vec_data_128 = _mm_loadu_ps(ptr + i);
        _mm_storeu_ps(ptr + i, _mm_add_ps(vec_data_128, vec_bias_128));
      }
#endif
      for(; i < channel; i++)
      {
        *(ptr + i) = pbias[i] + ptr[i];                    
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
