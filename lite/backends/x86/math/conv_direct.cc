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
#include "lite/backends/x86/math/conv_utils.h"
#include "lite/backends/x86/math/conv_direct.h"
#include "lite/core/context.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>
#ifdef __AVX__
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif
namespace paddle {
namespace lite {
namespace x86 {
namespace math {



#ifdef __AVX__

// From: https://stackoverflow.com/a/25627536
static inline void conv_direct_transpose8_ps(__m256& row0,  // NOLINT
                                 __m256& row1,  // NOLINT
                                 __m256& row2,  // NOLINT
                                 __m256& row3,  // NOLINT
                                 __m256& row4,  // NOLINT
                                 __m256& row5,  // NOLINT
                                 __m256& row6,  // NOLINT
                                 __m256& row7   // NOLINT
                                 ) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(row0, row1);
  __t1 = _mm256_unpackhi_ps(row0, row1);
  __t2 = _mm256_unpacklo_ps(row2, row3);
  __t3 = _mm256_unpackhi_ps(row2, row3);
  __t4 = _mm256_unpacklo_ps(row4, row5);
  __t5 = _mm256_unpackhi_ps(row4, row5);
  __t6 = _mm256_unpacklo_ps(row6, row7);
  __t7 = _mm256_unpackhi_ps(row6, row7);
  __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
  __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
  __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
  __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
  __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
  __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
  __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
  __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
  row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}
#else
static inline void conv_direct_transpose4_ps(__m128& row0,  // NOLINT
                                 __m128& row1,  // NOLINT
                                 __m128& row2,  // NOLINT
                                 __m128& row3  // NOLINT
                                 ) {
  __m128 __t0, __t1, __t2, __t3;
	__t0 = _mm_unpacklo_ps(row0, row1);
  __t1 = _mm_unpackhi_ps(row0, row1);
  __t2 = _mm_unpacklo_ps(row2, row3);
  __t3 = _mm_unpackhi_ps(row2, row3);
	row0 = _mm_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1, 0, 1, 0));
	row1 = _mm_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3, 2, 3, 2));
	row2 = _mm_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1, 0, 1, 0));
	row3 = _mm_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3, 2, 3, 2));
}
#endif

/* tranpose [chout, chin, wh, ww] to [chout/BLOCK,chin,wh,ww,BLOCK] */
void conv_trans_weights_numc(const float* din,
                             float* dout, // dout has been expanded
                             int chout,int chin,
                             int wh, int ww, int block) {

    // dout is [chout_expand / block , chin, wh, ww, block]
    int chout_expand = (chout + block -1) / block * block;
    memset(dout, 0.f, sizeof(float)  * chout_expand * chin * wh * ww);

    int w_index = -1;
    for(int wn_i = 0; wn_i < chout; wn_i ++)
    {
      for (int wc_i = 0; wc_i < chin; wc_i ++)
      {
        for (int wh_i = 0; wh_i < wh; wh_i ++)
        {
          for (int ww_i = 0; ww_i < ww; ww_i ++)
          {
            w_index ++;
            int dst_index = wn_i / block * chin * wh * ww * block+ wc_i * wh * ww * block  + wh_i * ww * block + ww_i * block + wn_i % block;  
            dout[dst_index] = din[w_index];
          }
        }
      }
    }
}

/* tranpose [chout,chin,wh,ww] to [chout/block,wh,ww,chin,block] */
void conv_trans_weights_forcin3(const float* din,
                                float* dout, // dout has been expanded
                                int chout,int chin,
                                int wh, int ww, int block) {

    //CHECK_EQ(chin, 3);
    int chout_expand = (chout + block -1) / block * block;
    memset(dout, 0.f, sizeof(float)  * chout_expand / block * wh * ww * chin * block);

    int w_index = -1;
    for(int wn_i = 0; wn_i < chout; wn_i ++)
    {
      for (int wc_i = 0; wc_i < chin; wc_i ++) // chin=3!
      {
        for (int wh_i = 0; wh_i < wh; wh_i ++)
        {
          for (int ww_i = 0; ww_i < ww; ww_i ++)
          {
            w_index ++;
            int dst_index = wn_i / block * wh * ww * chin * block + wh_i * ww * chin * block  + ww_i * chin * block + wc_i * block + wn_i % block;  
            dout[dst_index] = din[w_index];
          }
        }
      }
    }
}

/* 不就是NCHW-> NHWC吗！搞得那么神神秘秘*/
/* tranpose [bs,chin,ih,iw] to [bs,ih,iw,chin] */
void conv_trans_input_forcin3(const float* indata,
                              float* outdata,
                              int bs, int chin, int ih, int iw) {

    CHECK_EQ(chin, 3);
    memset(outdata, 0, sizeof(float)  * bs * ih * iw * chin);
    int w_index = -1;
    for(int bs_i = 0; bs_i < bs; bs_i ++)
    {
      for (int ic_i = 0; ic_i < chin; ic_i ++) // chin=3
      {
        for (int ih_i = 0; ih_i < ih; ih_i ++)
        {
          for (int iw_i = 0; iw_i < iw; iw_i ++)
          {
            w_index ++;
            int dst_index = bs_i * ih * iw * chin + ih_i * iw * chin  + iw_i * chin + ic_i;  
            outdata[dst_index] = indata[w_index];
          }
        }
      }
    }
}

/* trans_weight is [oc_expand / BLOCK, wh, ww, 3, BLOCK]*/
void conv_direct_3x3s2_forcin3_m256(const float* i_data,
                                    float* trans_i_data, // holds the tranposed input  
                                    const float* trans_weight,
                                    float* trans_out,  // holds the intermediate output result  
                                    int bs, int ic, int ih, int iw,
                                    int oc, int oc_expand, float* o_data,
                                    int oh, int ow, int ph, int pw)
{
  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int strideh = 2;
  constexpr int stridew = 2;

#ifdef __AVX__
  constexpr int BLOCK  = 8;
  constexpr int window_h = 5; //the sliding window is 5x7 and can obtain 2x3 results！ for AVX
  constexpr int window_w = 7;

#else
  constexpr int BLOCK  = 4;
  constexpr int window_h = 3;
  constexpr int window_w = 3;
#endif

double time1;
double time2;

  int wc = ic;
  time1 = (double)clock() / CLOCKS_PER_SEC;

  for (int bs_i = 0; bs_i < bs; bs_i ++) // 拿到第 bs_i个输入图片
  {
    memset(trans_out, 0, sizeof(float)  * oc_expand / BLOCK * oh * ow * BLOCK);
    conv_trans_input_forcin3(&i_data[bs_i * ic * ih * iw], trans_i_data, 1, ic, ih, iw);
    
    time2 = (double)clock() / CLOCKS_PER_SEC;
    std::cout << "转换输入参数 ms:"<<(time2 - time1) *1000 << std::endl;
    // 上面已经获得了输入的transpose！

    for (int group_i = 0; group_i < oc_expand / BLOCK; group_i ++)// 拿到第group_i组卷积核,自然要处理它的第ic_i个channel
    {
      // 上面已经获得了一个输入是三通道的，还有8个卷积核！然后这8个卷积核必然也是三通道的
      // 然后上面这些东西开始卷啦！
        int input_start_index = 0; // 从这个地方开始取出来三层吧！
        int kernel_start_index = group_i * wh * ww * wc * BLOCK;
        int output_start_index = group_i * oh * ow * BLOCK;

        
        int new_ih; // new_ih  
        int new_iw; // new_iw 表示滑窗左上角的最大数值
        int new_ih_start;
        int new_iw_start;
        if (ph == 0 && pw == 0)
        {
            new_iw = (iw - window_w) / 6 * 6; // 6表示w方向滑窗的步伐
            new_ih = (ih - window_h) / 4 * 4; // 4表示h方向滑窗的步伐
            new_ih_start = 0;
            new_iw_start = 0;
        }
        else if(ph == 1 && pw == 1)
        {
            new_iw = (iw - window_w - 1) / 6 * 6 + 1; // 6表示w方向滑窗的步伐
            new_ih = (ih - window_h - 1) / 4 * 4 + 1; // 4表示h方向滑窗的步伐
            new_ih_start = 1;
            new_iw_start = 1;
        }

        int o_left = (new_iw_start + pw) / 2;  //[0,o_left) 需特殊
        int o_right = (new_iw + pw) / 2 + 3;       // [o_right, ow)需特殊
        int o_upper = (new_ih_start + ph) / 2; //[0,o_upper) 需特殊
        int o_down = (new_ih + ph) / 2 + 2;        // [o_down, oh)需特殊

        for (int oh_i = 0; oh_i < o_upper; oh_i ++)
        {
            for (int ow_i = 0; ow_i < ow; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                // oh_i和ow_i是输出的索引，下面计算他们对应的输入的索引。
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;// 这俩个是对应的输入的左上角的哦！
                // 下面开始3x3的卷积吧！
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
                }
                _mm256_storeu_ps(&trans_out[output_index], res);      
            }
        }

        for (int oh_i = o_down; oh_i < oh; oh_i ++)
        {
            for (int ow_i = 0; ow_i < ow; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                // oh_i和ow_i是输出的索引，下面计算他们对应的输入的索引。
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;// 这俩个是对应的输入的左上角的哦！

                // 下面开始3x3的卷积吧！
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
                }
                _mm256_storeu_ps(&trans_out[output_index], res);
            }
        }

        for (int oh_i = 0; oh_i < oh; oh_i ++)
        {
            if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
                continue;
            for (int ow_i = 0; ow_i < o_left; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                // oh_i和ow_i是输出的索引，下面计算他们对应的输入的索引。
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;// 这俩个是对应的输入的左上角的哦！
                // 下面开始3x3的卷积吧！
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
                }
                _mm256_storeu_ps(&trans_out[output_index], res);
            }
        }

        for (int oh_i = 0; oh_i < oh; oh_i ++)
        {
            if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
                continue;
            for (int ow_i = o_right; ow_i < ow; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                // oh_i和ow_i是输出的索引，下面计算他们对应的输入的索引。
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;// 这俩个是对应的输入的左上角的哦！

                // 下面开始3x3的卷积吧！
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
                }
                _mm256_storeu_ps(&trans_out[output_index], res);
            }
        }

        for (int ih_i = new_ih_start; ih_i <= new_ih; ih_i += 4)
        {
          for (int iw_i = new_iw_start; iw_i <= new_iw; iw_i += 6)
          {
            int output_index = output_start_index + (ih_i + ph) / 2 * ow * BLOCK + (iw_i + pw) / 2 * BLOCK;

/* 下面是naive的方法1*/
/*
// 下面首先取出3个第一层的卷积核
// 每一层都有3个通道
            __m256 w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 0 * BLOCK]);
            __m256 w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 1 * BLOCK]);
            __m256 w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 2 * BLOCK]);  
            __m256 w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 3 * BLOCK]);
            __m256 w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 4 * BLOCK]);
            __m256 w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 5 * BLOCK]);
            __m256 w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 6 * BLOCK]);
            __m256 w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 7 * BLOCK]);
            __m256 w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 8 * BLOCK]);

            __m256 w100 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 9 * BLOCK]);
            __m256 w101 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 10 * BLOCK]);
            __m256 w102 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 11 * BLOCK]);  
            __m256 w110 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 12 * BLOCK]);
            __m256 w111 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 13 * BLOCK]);
            __m256 w112 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 14 * BLOCK]);
            __m256 w120 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 15 * BLOCK]);
            __m256 w121 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 16 * BLOCK]);
            __m256 w122 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 17 * BLOCK]);
            
            __m256 w200 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 18 * BLOCK]);
            __m256 w201 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 19 * BLOCK]);
            __m256 w202 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 20 * BLOCK]);  
            __m256 w210 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 21 * BLOCK]);
            __m256 w211 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 22 * BLOCK]);
            __m256 w212 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 23 * BLOCK]);
            __m256 w220 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 24 * BLOCK]);
            __m256 w221 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 25 * BLOCK]);
            __m256 w222 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 26 * BLOCK]);

// 下面开始取出来5行，每行是我要用(012 234 456)*3个数字！
const  float* iv0 = trans_i_data + input_start_index + ih_i * iw * ic + iw_i * ic;
const  float* iv1 = iv0 + 1 * iw * ic;
const  float* iv2 = iv0 + 2 * iw * ic;
const  float* iv3 = iv0 + 3 * iw * ic;
const  float* iv4 = iv0 + 4 * iw * ic;

// 一共可产生6个结果

__m256 res00 = w000 * _mm256_set1_ps(iv0[0]) + w001 * _mm256_set1_ps(iv0[1]) + w002 * _mm256_set1_ps(iv0[2]) + 
        w010 * _mm256_set1_ps(iv0[3]) + w011 * _mm256_set1_ps(iv0[4]) + w012 * _mm256_set1_ps(iv0[5]) + 
        w020 * _mm256_set1_ps(iv0[6]) + w021 * _mm256_set1_ps(iv0[7]) + w022 * _mm256_set1_ps(iv0[8]) +
        w100 * _mm256_set1_ps(iv1[0]) + w101 * _mm256_set1_ps(iv1[1]) + w102 * _mm256_set1_ps(iv1[2]) + 
        w110 * _mm256_set1_ps(iv1[3]) + w111 * _mm256_set1_ps(iv1[4]) + w112 * _mm256_set1_ps(iv1[5]) + 
        w120 * _mm256_set1_ps(iv1[6]) + w121 * _mm256_set1_ps(iv1[7]) + w122 * _mm256_set1_ps(iv1[8]) +
        w200 * _mm256_set1_ps(iv2[0]) + w201 * _mm256_set1_ps(iv2[1]) + w202 * _mm256_set1_ps(iv2[2]) + 
        w210 * _mm256_set1_ps(iv2[3]) + w211 * _mm256_set1_ps(iv2[4]) + w212 * _mm256_set1_ps(iv2[5]) + 
        w220 * _mm256_set1_ps(iv2[6]) + w221 * _mm256_set1_ps(iv2[7]) + w222 * _mm256_set1_ps(iv2[8]);

__m256 res01 = w000 * _mm256_set1_ps(iv0[6]) + w001 * _mm256_set1_ps(iv0[7]) + w002 * _mm256_set1_ps(iv0[8]) + 
        w010 * _mm256_set1_ps(iv0[9]) + w011 * _mm256_set1_ps(iv0[10]) + w012 * _mm256_set1_ps(iv0[11]) + 
        w020 * _mm256_set1_ps(iv0[12]) + w021 * _mm256_set1_ps(iv0[13]) + w022 * _mm256_set1_ps(iv0[14]) +
        w100 * _mm256_set1_ps(iv1[6]) + w101 * _mm256_set1_ps(iv1[7]) + w102 * _mm256_set1_ps(iv1[8]) + 
        w110 * _mm256_set1_ps(iv1[9]) + w111 * _mm256_set1_ps(iv1[10]) + w112 * _mm256_set1_ps(iv1[11]) + 
        w120 * _mm256_set1_ps(iv1[12]) + w121 * _mm256_set1_ps(iv1[13]) + w122 * _mm256_set1_ps(iv1[14]) +
        w200 * _mm256_set1_ps(iv2[6]) + w201 * _mm256_set1_ps(iv2[7]) + w202 * _mm256_set1_ps(iv2[8]) + 
        w210 * _mm256_set1_ps(iv2[9]) + w211 * _mm256_set1_ps(iv2[10]) + w212 * _mm256_set1_ps(iv2[11]) + 
        w220 * _mm256_set1_ps(iv2[12]) + w221 * _mm256_set1_ps(iv2[13]) + w222 * _mm256_set1_ps(iv2[14]);

__m256 res02 = w000 * _mm256_set1_ps(iv0[12]) + w001 * _mm256_set1_ps(iv0[13]) + w002 * _mm256_set1_ps(iv0[14]) + 
        w010 * _mm256_set1_ps(iv0[15]) + w011 * _mm256_set1_ps(iv0[16]) + w012 * _mm256_set1_ps(iv0[17]) + 
        w020 * _mm256_set1_ps(iv0[18]) + w021 * _mm256_set1_ps(iv0[19]) + w022 * _mm256_set1_ps(iv0[20]) +
        w100 * _mm256_set1_ps(iv1[12]) + w101 * _mm256_set1_ps(iv1[13]) + w102 * _mm256_set1_ps(iv1[14]) + 
        w110 * _mm256_set1_ps(iv1[15]) + w111 * _mm256_set1_ps(iv1[16]) + w112 * _mm256_set1_ps(iv1[17]) + 
        w120 * _mm256_set1_ps(iv1[18]) + w121 * _mm256_set1_ps(iv1[19]) + w122 * _mm256_set1_ps(iv1[20]) +
        w200 * _mm256_set1_ps(iv2[12]) + w201 * _mm256_set1_ps(iv2[13]) + w202 * _mm256_set1_ps(iv2[14]) + 
        w210 * _mm256_set1_ps(iv2[15]) + w211 * _mm256_set1_ps(iv2[16]) + w212 * _mm256_set1_ps(iv2[17]) + 
        w220 * _mm256_set1_ps(iv2[18]) + w221 * _mm256_set1_ps(iv2[19]) + w222 * _mm256_set1_ps(iv2[20]);

__m256 res10 = w000 * _mm256_set1_ps(iv2[0]) + w001 * _mm256_set1_ps(iv2[1]) + w002 * _mm256_set1_ps(iv2[2]) + 
        w010 * _mm256_set1_ps(iv2[3]) + w011 * _mm256_set1_ps(iv2[4]) + w012 * _mm256_set1_ps(iv2[5]) + 
        w020 * _mm256_set1_ps(iv2[6]) + w021 * _mm256_set1_ps(iv2[7]) + w022 * _mm256_set1_ps(iv2[8]) +
        w100 * _mm256_set1_ps(iv3[0]) + w101 * _mm256_set1_ps(iv3[1]) + w102 * _mm256_set1_ps(iv3[2]) + 
        w110 * _mm256_set1_ps(iv3[3]) + w111 * _mm256_set1_ps(iv3[4]) + w112 * _mm256_set1_ps(iv3[5]) + 
        w120 * _mm256_set1_ps(iv3[6]) + w121 * _mm256_set1_ps(iv3[7]) + w122 * _mm256_set1_ps(iv3[8]) +
        w200 * _mm256_set1_ps(iv4[0]) + w201 * _mm256_set1_ps(iv4[1]) + w202 * _mm256_set1_ps(iv4[2]) + 
        w210 * _mm256_set1_ps(iv4[3]) + w211 * _mm256_set1_ps(iv4[4]) + w212 * _mm256_set1_ps(iv4[5]) + 
        w220 * _mm256_set1_ps(iv4[6]) + w221 * _mm256_set1_ps(iv4[7]) + w222 * _mm256_set1_ps(iv4[8]);

__m256 res11 = w000 * _mm256_set1_ps(iv2[6]) + w001 * _mm256_set1_ps(iv2[7]) + w002 * _mm256_set1_ps(iv2[8]) + 
        w010 * _mm256_set1_ps(iv2[9]) + w011 * _mm256_set1_ps(iv2[10]) + w012 * _mm256_set1_ps(iv2[11]) + 
        w020 * _mm256_set1_ps(iv2[12]) + w021 * _mm256_set1_ps(iv2[13]) + w022 * _mm256_set1_ps(iv2[14]) +
        w100 * _mm256_set1_ps(iv3[6]) + w101 * _mm256_set1_ps(iv3[7]) + w102 * _mm256_set1_ps(iv3[8]) + 
        w110 * _mm256_set1_ps(iv3[9]) + w111 * _mm256_set1_ps(iv3[10]) + w112 * _mm256_set1_ps(iv3[11]) + 
        w120 * _mm256_set1_ps(iv3[12]) + w121 * _mm256_set1_ps(iv3[13]) + w122 * _mm256_set1_ps(iv3[14]) +
        w200 * _mm256_set1_ps(iv4[6]) + w201 * _mm256_set1_ps(iv4[7]) + w202 * _mm256_set1_ps(iv4[8]) + 
        w210 * _mm256_set1_ps(iv4[9]) + w211 * _mm256_set1_ps(iv4[10]) + w212 * _mm256_set1_ps(iv4[11]) + 
        w220 * _mm256_set1_ps(iv4[12]) + w221 * _mm256_set1_ps(iv4[13]) + w222 * _mm256_set1_ps(iv4[14]);

__m256 res12 = w000 * _mm256_set1_ps(iv2[12]) + w001 * _mm256_set1_ps(iv2[13]) + w002 * _mm256_set1_ps(iv2[14]) + 
        w010 * _mm256_set1_ps(iv2[15]) + w011 * _mm256_set1_ps(iv2[16]) + w012 * _mm256_set1_ps(iv2[17]) + 
        w020 * _mm256_set1_ps(iv2[18]) + w021 * _mm256_set1_ps(iv2[19]) + w022 * _mm256_set1_ps(iv2[20]) +
        w100 * _mm256_set1_ps(iv3[12]) + w101 * _mm256_set1_ps(iv3[13]) + w102 * _mm256_set1_ps(iv3[14]) + 
        w110 * _mm256_set1_ps(iv3[15]) + w111 * _mm256_set1_ps(iv3[16]) + w112 * _mm256_set1_ps(iv3[17]) + 
        w120 * _mm256_set1_ps(iv3[18]) + w121 * _mm256_set1_ps(iv3[19]) + w122 * _mm256_set1_ps(iv3[20]) +
        w200 * _mm256_set1_ps(iv4[12]) + w201 * _mm256_set1_ps(iv4[13]) + w202 * _mm256_set1_ps(iv4[14]) + 
        w210 * _mm256_set1_ps(iv4[15]) + w211 * _mm256_set1_ps(iv4[16]) + w212 * _mm256_set1_ps(iv4[17]) + 
        w220 * _mm256_set1_ps(iv4[18]) + w221 * _mm256_set1_ps(iv4[19]) + w222 * _mm256_set1_ps(iv4[20]);

_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res10);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res11);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res12);
*/
/*naive 方法结束*/

/*下面是第二种方法，这个方法一次性把卷积核的参数全部取出来了*/
/*
            // 下面我要先取出卷积核的第一行的9个参数吧！
            __m256 w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 0 * BLOCK]);
            __m256 w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 1 * BLOCK]);
            __m256 w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 2 * BLOCK]);
            __m256 w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 3 * BLOCK]);
            __m256 w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 4 * BLOCK]);
            __m256 w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 5 * BLOCK]);  
            __m256 w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 6 * BLOCK]);
            __m256 w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 7 * BLOCK]);
            __m256 w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 8 * BLOCK]);

            __m256 w100 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 9 * BLOCK]);
            __m256 w101 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 10 * BLOCK]);
            __m256 w102 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 11 * BLOCK]);  
            __m256 w110 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 12 * BLOCK]);
            __m256 w111 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 13 * BLOCK]);
            __m256 w112 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 14 * BLOCK]);
            __m256 w120 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 15 * BLOCK]);
            __m256 w121 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 16 * BLOCK]);
            __m256 w122 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 17 * BLOCK]);
            
            __m256 w200 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 18 * BLOCK]);
            __m256 w201 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 19 * BLOCK]);
            __m256 w202 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 20 * BLOCK]);  
            __m256 w210 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 21 * BLOCK]);
            __m256 w211 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 22 * BLOCK]);
            __m256 w212 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 23 * BLOCK]);
            __m256 w220 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 24 * BLOCK]);
            __m256 w221 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 25 * BLOCK]);
            __m256 w222 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 26 * BLOCK]);

// 下面开始取出来24个数字
const  float* iv0 = trans_i_data + input_start_index + ih_i * iw * ic + iw_i * ic;
const  float* iv1 = iv0 + 1 * iw * ic;
const  float* iv2 = iv0 + 2 * iw * ic;
const  float* iv3 = iv0 + 3 * iw * ic;
const  float* iv4 = iv0 + 4 * iw * ic;

// 下面那就取出来res

// 下面接着拿出ivo行的第0，1，2个数
// 第3，4，5个数
// 第6，7，8,这
__m256 input0 = _mm256_set1_ps(iv0[0]); // 这三个我打算复用呢  
__m256 input1 = _mm256_set1_ps(iv0[1]);
__m256 input2 = _mm256_set1_ps(iv0[2]);
__m256 res00 = input0 * w000;
__m256 res01 = input1 * w001;
__m256 res02 = input2 * w002;
input0 = _mm256_set1_ps(iv0[3]);  
input1 = _mm256_set1_ps(iv0[4]);
input2 = _mm256_set1_ps(iv0[5]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv0[6]);  
input1 = _mm256_set1_ps(iv0[7]);
input2 = _mm256_set1_ps(iv0[8]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
res00 = res00 + res01 + res02;
// 第二行
input0 = _mm256_set1_ps(iv1[0]);  
input1 = _mm256_set1_ps(iv1[1]);
input2 = _mm256_set1_ps(iv1[2]);
res00 = _mm256_fmadd_ps(input0, w100, res00);
res00 = _mm256_fmadd_ps(input1, w101, res00);
res00 = _mm256_fmadd_ps(input2, w102, res00);
input0 = _mm256_set1_ps(iv1[3]);  
input1 = _mm256_set1_ps(iv1[4]);
input2 = _mm256_set1_ps(iv1[5]);
res00 = _mm256_fmadd_ps(input0, w110, res00);
res00 = _mm256_fmadd_ps(input1, w111, res00);
res00 = _mm256_fmadd_ps(input2, w112, res00);
input0 = _mm256_set1_ps(iv1[6]);  
input1 = _mm256_set1_ps(iv1[7]);
input2 = _mm256_set1_ps(iv1[8]);
res00 = _mm256_fmadd_ps(input0, w120, res00);
res00 = _mm256_fmadd_ps(input1, w121, res00);
res00 = _mm256_fmadd_ps(input2, w122, res00);
// 第三行
input0 = _mm256_set1_ps(iv2[0]);  
input1 = _mm256_set1_ps(iv2[1]);
input2 = _mm256_set1_ps(iv2[2]);
res00 = _mm256_fmadd_ps(input0, w200, res00);
res00 = _mm256_fmadd_ps(input1, w201, res00);
res00 = _mm256_fmadd_ps(input2, w202, res00);
input0 = _mm256_set1_ps(iv2[3]);  
input1 = _mm256_set1_ps(iv2[4]);
input2 = _mm256_set1_ps(iv2[5]);
res00 = _mm256_fmadd_ps(input0, w210, res00);
res00 = _mm256_fmadd_ps(input1, w211, res00);
res00 = _mm256_fmadd_ps(input2, w212, res00);
input0 = _mm256_set1_ps(iv2[6]);  
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[8]);
res00 = _mm256_fmadd_ps(input0, w220, res00);
res00 = _mm256_fmadd_ps(input1, w221, res00);
res00 = _mm256_fmadd_ps(input2, w222, res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);

// 第二个结果
input0 = _mm256_set1_ps(iv0[6]);
input1 = _mm256_set1_ps(iv0[7]);
input2 = _mm256_set1_ps(iv0[8]);
res00 = input0 * w000;
res00 = _mm256_fmadd_ps(input1, w001, res00);
res00 = _mm256_fmadd_ps(input2, w002, res00);
input0 = _mm256_set1_ps(iv0[9]);  
input1 = _mm256_set1_ps(iv0[10]);
input2 = _mm256_set1_ps(iv0[11]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res00 = _mm256_fmadd_ps(input1, w011, res00);
res00 = _mm256_fmadd_ps(input2, w012, res00);
input0 = _mm256_set1_ps(iv0[12]);  
input1 = _mm256_set1_ps(iv0[13]);
input2 = _mm256_set1_ps(iv0[14]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res00 = _mm256_fmadd_ps(input1, w021, res00);
res00 = _mm256_fmadd_ps(input2, w022, res00);
// 第二行
input0 = _mm256_set1_ps(iv1[6]);  
input1 = _mm256_set1_ps(iv1[7]);
input2 = _mm256_set1_ps(iv1[8]);
res00 = _mm256_fmadd_ps(input0, w100, res00);
res00 = _mm256_fmadd_ps(input1, w101, res00);
res00 = _mm256_fmadd_ps(input2, w102, res00);
input0 = _mm256_set1_ps(iv1[9]);  
input1 = _mm256_set1_ps(iv1[10]);
input2 = _mm256_set1_ps(iv1[11]);
res00 = _mm256_fmadd_ps(input0, w110, res00);
res00 = _mm256_fmadd_ps(input1, w111, res00);
res00 = _mm256_fmadd_ps(input2, w112, res00);
input0 = _mm256_set1_ps(iv1[12]);  
input1 = _mm256_set1_ps(iv1[13]);
input2 = _mm256_set1_ps(iv1[14]);
res00 = _mm256_fmadd_ps(input0, w120, res00);
res00 = _mm256_fmadd_ps(input1, w121, res00);
res00 = _mm256_fmadd_ps(input2, w122, res00);
// 第三行
input0 = _mm256_set1_ps(iv2[6]);  
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[8]);
res00 = _mm256_fmadd_ps(input0, w200, res00);
res00 = _mm256_fmadd_ps(input1, w201, res00);
res00 = _mm256_fmadd_ps(input2, w202, res00);
input0 = _mm256_set1_ps(iv2[9]);  
input1 = _mm256_set1_ps(iv2[10]);
input2 = _mm256_set1_ps(iv2[11]);
res00 = _mm256_fmadd_ps(input0, w210, res00);
res00 = _mm256_fmadd_ps(input1, w211, res00);
res00 = _mm256_fmadd_ps(input2, w212, res00);
input0 = _mm256_set1_ps(iv2[12]);  
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w220, res00);
res00 = _mm256_fmadd_ps(input1, w221, res00);
res00 = _mm256_fmadd_ps(input2, w222, res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res00);
// 第三个结果
input0 = _mm256_set1_ps(iv0[12]);
input1 = _mm256_set1_ps(iv0[13]);
input2 = _mm256_set1_ps(iv0[14]);
res00 = input0 * w000;
res00 = _mm256_fmadd_ps(input1, w001, res00);
res00 = _mm256_fmadd_ps(input2, w002, res00);
input0 = _mm256_set1_ps(iv0[15]);  
input1 = _mm256_set1_ps(iv0[16]);
input2 = _mm256_set1_ps(iv0[17]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res00 = _mm256_fmadd_ps(input1, w011, res00);
res00 = _mm256_fmadd_ps(input2, w012, res00);
input0 = _mm256_set1_ps(iv0[18]);  
input1 = _mm256_set1_ps(iv0[19]);
input2 = _mm256_set1_ps(iv0[20]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res00 = _mm256_fmadd_ps(input1, w021, res00);
res00 = _mm256_fmadd_ps(input2, w022, res00);
// 第二行
input0 = _mm256_set1_ps(iv1[12]);  
input1 = _mm256_set1_ps(iv1[13]);
input2 = _mm256_set1_ps(iv1[14]);
res00 = _mm256_fmadd_ps(input0, w100, res00);
res00 = _mm256_fmadd_ps(input1, w101, res00);
res00 = _mm256_fmadd_ps(input2, w102, res00);
input0 = _mm256_set1_ps(iv1[15]);  
input1 = _mm256_set1_ps(iv1[16]);
input2 = _mm256_set1_ps(iv1[17]);
res00 = _mm256_fmadd_ps(input0, w110, res00);
res00 = _mm256_fmadd_ps(input1, w111, res00);
res00 = _mm256_fmadd_ps(input2, w112, res00);
input0 = _mm256_set1_ps(iv1[18]);  
input1 = _mm256_set1_ps(iv1[19]);
input2 = _mm256_set1_ps(iv1[20]);
res00 = _mm256_fmadd_ps(input0, w120, res00);
res00 = _mm256_fmadd_ps(input1, w121, res00);
res00 = _mm256_fmadd_ps(input2, w122, res00);
// 第三行
input0 = _mm256_set1_ps(iv2[12]);  
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w200, res00);
res00 = _mm256_fmadd_ps(input1, w201, res00);
res00 = _mm256_fmadd_ps(input2, w202, res00);
input0 = _mm256_set1_ps(iv2[15]);  
input1 = _mm256_set1_ps(iv2[16]);
input2 = _mm256_set1_ps(iv2[17]);
res00 = _mm256_fmadd_ps(input0, w210, res00);
res00 = _mm256_fmadd_ps(input1, w211, res00);
res00 = _mm256_fmadd_ps(input2, w212, res00);
input0 = _mm256_set1_ps(iv2[18]);  
input1 = _mm256_set1_ps(iv2[19]);
input2 = _mm256_set1_ps(iv2[20]);
res00 = _mm256_fmadd_ps(input0, w220, res00);
res00 = _mm256_fmadd_ps(input1, w221, res00);
res00 = _mm256_fmadd_ps(input2, w222, res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res00);

// 第二行第1个结果
input0 = _mm256_set1_ps(iv2[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv2[1]);
input2 = _mm256_set1_ps(iv2[2]);
res00 = input0 * w000;
res00 = _mm256_fmadd_ps(input1, w001, res00);
res00 = _mm256_fmadd_ps(input2, w002, res00);
input0 = _mm256_set1_ps(iv2[3]);  
input1 = _mm256_set1_ps(iv2[4]);
input2 = _mm256_set1_ps(iv2[5]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res00 = _mm256_fmadd_ps(input1, w011, res00);
res00 = _mm256_fmadd_ps(input2, w012, res00);
input0 = _mm256_set1_ps(iv2[6]);  
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[8]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res00 = _mm256_fmadd_ps(input1, w021, res00);
res00 = _mm256_fmadd_ps(input2, w022, res00);
// 第二行
input0 = _mm256_set1_ps(iv3[0]);  
input1 = _mm256_set1_ps(iv3[1]);
input2 = _mm256_set1_ps(iv3[2]);
res00 = _mm256_fmadd_ps(input0, w100, res00);
res00 = _mm256_fmadd_ps(input1, w101, res00);
res00 = _mm256_fmadd_ps(input2, w102, res00);
input0 = _mm256_set1_ps(iv3[3]);  
input1 = _mm256_set1_ps(iv3[4]);
input2 = _mm256_set1_ps(iv3[5]);
res00 = _mm256_fmadd_ps(input0, w110, res00);
res00 = _mm256_fmadd_ps(input1, w111, res00);
res00 = _mm256_fmadd_ps(input2, w112, res00);
input0 = _mm256_set1_ps(iv3[6]);  
input1 = _mm256_set1_ps(iv3[7]);
input2 = _mm256_set1_ps(iv3[8]);
res00 = _mm256_fmadd_ps(input0, w120, res00);
res00 = _mm256_fmadd_ps(input1, w121, res00);
res00 = _mm256_fmadd_ps(input2, w122, res00);
// 第三行
input0 = _mm256_set1_ps(iv4[0]);  
input1 = _mm256_set1_ps(iv4[1]);
input2 = _mm256_set1_ps(iv4[2]);
res00 = _mm256_fmadd_ps(input0, w200, res00);
res00 = _mm256_fmadd_ps(input1, w201, res00);
res00 = _mm256_fmadd_ps(input2, w202, res00);
input0 = _mm256_set1_ps(iv4[3]);  
input1 = _mm256_set1_ps(iv4[4]);
input2 = _mm256_set1_ps(iv4[5]);
res00 = _mm256_fmadd_ps(input0, w210, res00);
res00 = _mm256_fmadd_ps(input1, w211, res00);
res00 = _mm256_fmadd_ps(input2, w212, res00);
input0 = _mm256_set1_ps(iv4[6]);  
input1 = _mm256_set1_ps(iv4[7]);
input2 = _mm256_set1_ps(iv4[8]);
res00 = _mm256_fmadd_ps(input0, w220, res00);
res00 = _mm256_fmadd_ps(input1, w221, res00);
res00 = _mm256_fmadd_ps(input2, w222, res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
// 第二行第二个结果
input0 = _mm256_set1_ps(iv2[6]);
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[8]);
res00 = input0 * w000;
res00 = _mm256_fmadd_ps(input1, w001, res00);
res00 = _mm256_fmadd_ps(input2, w002, res00);
input0 = _mm256_set1_ps(iv2[9]);  
input1 = _mm256_set1_ps(iv2[10]);
input2 = _mm256_set1_ps(iv2[11]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res00 = _mm256_fmadd_ps(input1, w011, res00);
res00 = _mm256_fmadd_ps(input2, w012, res00);
input0 = _mm256_set1_ps(iv2[12]);  
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res00 = _mm256_fmadd_ps(input1, w021, res00);
res00 = _mm256_fmadd_ps(input2, w022, res00);
// 第二行
input0 = _mm256_set1_ps(iv3[6]);  
input1 = _mm256_set1_ps(iv3[7]);
input2 = _mm256_set1_ps(iv3[8]);
res00 = _mm256_fmadd_ps(input0, w100, res00);
res00 = _mm256_fmadd_ps(input1, w101, res00);
res00 = _mm256_fmadd_ps(input2, w102, res00);
input0 = _mm256_set1_ps(iv3[9]);  
input1 = _mm256_set1_ps(iv3[10]);
input2 = _mm256_set1_ps(iv3[11]);
res00 = _mm256_fmadd_ps(input0, w110, res00);
res00 = _mm256_fmadd_ps(input1, w111, res00);
res00 = _mm256_fmadd_ps(input2, w112, res00);
input0 = _mm256_set1_ps(iv3[12]);  
input1 = _mm256_set1_ps(iv3[13]);
input2 = _mm256_set1_ps(iv3[14]);
res00 = _mm256_fmadd_ps(input0, w120, res00);
res00 = _mm256_fmadd_ps(input1, w121, res00);
res00 = _mm256_fmadd_ps(input2, w122, res00);
// 第三行
input0 = _mm256_set1_ps(iv4[6]);  
input1 = _mm256_set1_ps(iv4[7]);
input2 = _mm256_set1_ps(iv4[8]);
res00 = _mm256_fmadd_ps(input0, w200, res00);
res00 = _mm256_fmadd_ps(input1, w201, res00);
res00 = _mm256_fmadd_ps(input2, w202, res00);
input0 = _mm256_set1_ps(iv4[9]);  
input1 = _mm256_set1_ps(iv4[10]);
input2 = _mm256_set1_ps(iv4[11]);
res00 = _mm256_fmadd_ps(input0, w210, res00);
res00 = _mm256_fmadd_ps(input1, w211, res00);
res00 = _mm256_fmadd_ps(input2, w212, res00);
input0 = _mm256_set1_ps(iv4[12]);  
input1 = _mm256_set1_ps(iv4[13]);
input2 = _mm256_set1_ps(iv4[14]);
res00 = _mm256_fmadd_ps(input0, w220, res00);
res00 = _mm256_fmadd_ps(input1, w221, res00);
res00 = _mm256_fmadd_ps(input2, w222, res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res00);
// 第二行第三个结果
input0 = _mm256_set1_ps(iv2[12]);
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = input0 * w000;
res00 = _mm256_fmadd_ps(input1, w001, res00);
res00 = _mm256_fmadd_ps(input2, w002, res00);
input0 = _mm256_set1_ps(iv2[15]);  
input1 = _mm256_set1_ps(iv2[16]);
input2 = _mm256_set1_ps(iv2[17]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res00 = _mm256_fmadd_ps(input1, w011, res00);
res00 = _mm256_fmadd_ps(input2, w012, res00);
input0 = _mm256_set1_ps(iv2[18]);  
input1 = _mm256_set1_ps(iv2[19]);
input2 = _mm256_set1_ps(iv2[20]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res00 = _mm256_fmadd_ps(input1, w021, res00);
res00 = _mm256_fmadd_ps(input2, w022, res00);
// 第二行
input0 = _mm256_set1_ps(iv3[12]);  
input1 = _mm256_set1_ps(iv3[13]);
input2 = _mm256_set1_ps(iv3[14]);
res00 = _mm256_fmadd_ps(input0, w100, res00);
res00 = _mm256_fmadd_ps(input1, w101, res00);
res00 = _mm256_fmadd_ps(input2, w102, res00);
input0 = _mm256_set1_ps(iv3[15]);  
input1 = _mm256_set1_ps(iv3[16]);
input2 = _mm256_set1_ps(iv3[17]);
res00 = _mm256_fmadd_ps(input0, w110, res00);
res00 = _mm256_fmadd_ps(input1, w111, res00);
res00 = _mm256_fmadd_ps(input2, w112, res00);
input0 = _mm256_set1_ps(iv3[18]);  
input1 = _mm256_set1_ps(iv3[19]);
input2 = _mm256_set1_ps(iv3[20]);
res00 = _mm256_fmadd_ps(input0, w120, res00);
res00 = _mm256_fmadd_ps(input1, w121, res00);
res00 = _mm256_fmadd_ps(input2, w122, res00);
// 第三行
input0 = _mm256_set1_ps(iv4[12]);  
input1 = _mm256_set1_ps(iv4[13]);
input2 = _mm256_set1_ps(iv4[14]);
res00 = _mm256_fmadd_ps(input0, w200, res00);
res00 = _mm256_fmadd_ps(input1, w201, res00);
res00 = _mm256_fmadd_ps(input2, w202, res00);
input0 = _mm256_set1_ps(iv4[15]);  
input1 = _mm256_set1_ps(iv4[16]);
input2 = _mm256_set1_ps(iv4[17]);
res00 = _mm256_fmadd_ps(input0, w210, res00);
res00 = _mm256_fmadd_ps(input1, w211, res00);
res00 = _mm256_fmadd_ps(input2, w212, res00);
input0 = _mm256_set1_ps(iv4[18]);  
input1 = _mm256_set1_ps(iv4[19]);
input2 = _mm256_set1_ps(iv4[20]);
res00 = _mm256_fmadd_ps(input0, w220, res00);
res00 = _mm256_fmadd_ps(input1, w221, res00);
res00 = _mm256_fmadd_ps(input2, w222, res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res00);
*/
/*第二种方法结束*/



/*下面是第333333333333种方法，这个方法一次性只把第一层的参数先取出来，然后用完了他再取第二层*/
/*
            // 先取出卷积核的第一行的9个参数吧！，然后复用这9个位置！
            __m256 w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 0 * BLOCK]);
            __m256 w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 1 * BLOCK]);
            __m256 w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 2 * BLOCK]);
            __m256 w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 3 * BLOCK]);
            __m256 w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 4 * BLOCK]);
            __m256 w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 5 * BLOCK]);  
            __m256 w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 6 * BLOCK]);
            __m256 w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 7 * BLOCK]);
            __m256 w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 8 * BLOCK]);

// 下面开始取出来24个数字
const  float* iv0 = trans_i_data + input_start_index + ih_i * iw * ic + iw_i * ic;
const  float* iv1 = iv0 + 1 * iw * ic;
const  float* iv2 = iv0 + 2 * iw * ic;
const  float* iv3 = iv0 + 3 * iw * ic;
const  float* iv4 = iv0 + 4 * iw * ic;

// 下面那就取出来res

// 下面接着拿出ivo行的第0，1，2个数
// 第3，4，5个数
// 第6，7，8,这
__m256 input0 = _mm256_set1_ps(iv0[0]); // 这三个我打算复用呢  
__m256 input1 = _mm256_set1_ps(iv0[6]);
__m256 input2 = _mm256_set1_ps(iv0[12]);
__m256 res00 = input0 * w000;
__m256 res01 = input1 * w000;
__m256 res02 = input2 * w000;
input0 = _mm256_set1_ps(iv0[1]); 
input1 = _mm256_set1_ps(iv0[7]);
input2 = _mm256_set1_ps(iv0[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv0[2]); 
input1 = _mm256_set1_ps(iv0[8]);
input2 = _mm256_set1_ps(iv0[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv0[3]); 
input1 = _mm256_set1_ps(iv0[9]);
input2 = _mm256_set1_ps(iv0[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv0[4]); 
input1 = _mm256_set1_ps(iv0[10]);
input2 = _mm256_set1_ps(iv0[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv0[5]); 
input1 = _mm256_set1_ps(iv0[11]);
input2 = _mm256_set1_ps(iv0[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv0[6]); 
input1 = _mm256_set1_ps(iv0[12]);
input2 = _mm256_set1_ps(iv0[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv0[7]); 
input1 = _mm256_set1_ps(iv0[13]);
input2 = _mm256_set1_ps(iv0[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv0[8]); 
input1 = _mm256_set1_ps(iv0[14]);
input2 = _mm256_set1_ps(iv0[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
// 下面将第一行的三层参数乘 iv2的东西
input0 = _mm256_set1_ps(iv2[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv2[6]);
input2 = _mm256_set1_ps(iv2[12]);
res00 = input0 * w000;
res01 = input1 * w000;
res02 = input2 * w000;
input0 = _mm256_set1_ps(iv2[1]); 
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv2[2]); 
input1 = _mm256_set1_ps(iv2[8]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv2[3]); 
input1 = _mm256_set1_ps(iv2[9]);
input2 = _mm256_set1_ps(iv2[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv2[4]); 
input1 = _mm256_set1_ps(iv2[10]);
input2 = _mm256_set1_ps(iv2[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv2[5]); 
input1 = _mm256_set1_ps(iv2[11]);
input2 = _mm256_set1_ps(iv2[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv2[6]); 
input1 = _mm256_set1_ps(iv2[12]);
input2 = _mm256_set1_ps(iv2[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv2[7]); 
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv2[8]); 
input1 = _mm256_set1_ps(iv2[14]);
input2 = _mm256_set1_ps(iv2[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

// 开始将第二行的三层参数load进来
w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 9 * BLOCK]);
w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 10 * BLOCK]);
w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 11 * BLOCK]);
w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 12 * BLOCK]);
w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 13 * BLOCK]);
w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 14 * BLOCK]);  
w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 15 * BLOCK]);
w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 16 * BLOCK]);
w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 17 * BLOCK]);
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2]);
input0 = _mm256_set1_ps(iv1[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv1[6]);
input2 = _mm256_set1_ps(iv1[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv1[1]); 
input1 = _mm256_set1_ps(iv1[7]);
input2 = _mm256_set1_ps(iv1[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv1[2]); 
input1 = _mm256_set1_ps(iv1[8]);
input2 = _mm256_set1_ps(iv1[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv1[3]); 
input1 = _mm256_set1_ps(iv1[9]);
input2 = _mm256_set1_ps(iv1[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv1[4]); 
input1 = _mm256_set1_ps(iv1[10]);
input2 = _mm256_set1_ps(iv1[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv1[5]); 
input1 = _mm256_set1_ps(iv1[11]);
input2 = _mm256_set1_ps(iv1[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv1[6]); 
input1 = _mm256_set1_ps(iv1[12]);
input2 = _mm256_set1_ps(iv1[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv1[7]); 
input1 = _mm256_set1_ps(iv1[13]);
input2 = _mm256_set1_ps(iv1[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv1[8]); 
input1 = _mm256_set1_ps(iv1[14]);
input2 = _mm256_set1_ps(iv1[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK]);
input0 = _mm256_set1_ps(iv3[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv3[6]);
input2 = _mm256_set1_ps(iv3[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv3[1]); 
input1 = _mm256_set1_ps(iv3[7]);
input2 = _mm256_set1_ps(iv3[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv3[2]); 
input1 = _mm256_set1_ps(iv3[8]);
input2 = _mm256_set1_ps(iv3[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv3[3]); 
input1 = _mm256_set1_ps(iv3[9]);
input2 = _mm256_set1_ps(iv3[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv3[4]); 
input1 = _mm256_set1_ps(iv3[10]);
input2 = _mm256_set1_ps(iv3[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv3[5]); 
input1 = _mm256_set1_ps(iv3[11]);
input2 = _mm256_set1_ps(iv3[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv3[6]); 
input1 = _mm256_set1_ps(iv3[12]);
input2 = _mm256_set1_ps(iv3[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv3[7]); 
input1 = _mm256_set1_ps(iv3[13]);
input2 = _mm256_set1_ps(iv3[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv3[8]); 
input1 = _mm256_set1_ps(iv3[14]);
input2 = _mm256_set1_ps(iv3[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

// 开始将第三行的三层参数load进来
w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 18 * BLOCK]);
w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 19 * BLOCK]);
w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 20 * BLOCK]);
w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 21 * BLOCK]);
w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 22 * BLOCK]);
w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 23 * BLOCK]);  
w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 24 * BLOCK]);
w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 25 * BLOCK]);
w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 26 * BLOCK]);
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2]);
input0 = _mm256_set1_ps(iv2[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv2[6]);
input2 = _mm256_set1_ps(iv2[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv2[1]); 
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv2[2]); 
input1 = _mm256_set1_ps(iv2[8]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv2[3]); 
input1 = _mm256_set1_ps(iv2[9]);
input2 = _mm256_set1_ps(iv2[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv2[4]); 
input1 = _mm256_set1_ps(iv2[10]);
input2 = _mm256_set1_ps(iv2[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv2[5]); 
input1 = _mm256_set1_ps(iv2[11]);
input2 = _mm256_set1_ps(iv2[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv2[6]); 
input1 = _mm256_set1_ps(iv2[12]);
input2 = _mm256_set1_ps(iv2[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv2[7]); 
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv2[8]); 
input1 = _mm256_set1_ps(iv2[14]);
input2 = _mm256_set1_ps(iv2[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK]);
input0 = _mm256_set1_ps(iv4[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv4[6]);
input2 = _mm256_set1_ps(iv4[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv4[1]); 
input1 = _mm256_set1_ps(iv4[7]);
input2 = _mm256_set1_ps(iv4[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv4[2]); 
input1 = _mm256_set1_ps(iv4[8]);
input2 = _mm256_set1_ps(iv4[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv4[3]); 
input1 = _mm256_set1_ps(iv4[9]);
input2 = _mm256_set1_ps(iv4[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv4[4]); 
input1 = _mm256_set1_ps(iv4[10]);
input2 = _mm256_set1_ps(iv4[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv4[5]); 
input1 = _mm256_set1_ps(iv4[11]);
input2 = _mm256_set1_ps(iv4[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv4[6]); 
input1 = _mm256_set1_ps(iv4[12]);
input2 = _mm256_set1_ps(iv4[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv4[7]); 
input1 = _mm256_set1_ps(iv4[13]);
input2 = _mm256_set1_ps(iv4[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv4[8]); 
input1 = _mm256_set1_ps(iv4[14]);
input2 = _mm256_set1_ps(iv4[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);

_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);
*/
/*第三种方法结束了*/

/*下面是第44444种方法，这个方法一次性只把第一层的参数先取出来，然后用完了他再取第二层*/

            // 先取出卷积核的第一行的9个参数吧！，然后复用这9个位置！
            __m256 w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 0 * BLOCK]);
            __m256 w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 1 * BLOCK]);
            __m256 w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 2 * BLOCK]);
            __m256 w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 3 * BLOCK]);
            __m256 w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 4 * BLOCK]);
            __m256 w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 5 * BLOCK]);  
            __m256 w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 6 * BLOCK]);
            __m256 w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 7 * BLOCK]);
            __m256 w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 8 * BLOCK]);

// 下面开始取出来24个数字
const  float* iv0 = trans_i_data + input_start_index + ih_i * iw * ic + iw_i * ic;
const  float* iv1 = iv0 + 1 * iw * ic;
const  float* iv2 = iv0 + 2 * iw * ic;
const  float* iv3 = iv0 + 3 * iw * ic;
const  float* iv4 = iv0 + 4 * iw * ic;

// 下面那就取出来res

// 下面接着拿出ivo行的第0，1，2个数
// 第3，4，5个数
// 第6，7，8,这
__m256 input0 = _mm256_set1_ps(iv0[0]); // 这三个我打算复用呢  
__m256 input1 = _mm256_set1_ps(iv0[6]);
__m256 input2 = _mm256_set1_ps(iv0[12]);
__m256 res00 = input0 * w000;
__m256 res01 = input1 * w000;
__m256 res02 = input2 * w000;
input0 = _mm256_set1_ps(iv0[1]); 
input1 = _mm256_set1_ps(iv0[7]);
input2 = _mm256_set1_ps(iv0[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv0[2]); 
input1 = _mm256_set1_ps(iv0[8]);
input2 = _mm256_set1_ps(iv0[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv0[3]); 
input1 = _mm256_set1_ps(iv0[9]);
input2 = _mm256_set1_ps(iv0[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv0[4]); 
input1 = _mm256_set1_ps(iv0[10]);
input2 = _mm256_set1_ps(iv0[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv0[5]); 
input1 = _mm256_set1_ps(iv0[11]);
input2 = _mm256_set1_ps(iv0[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv0[6]); 
input1 = _mm256_set1_ps(iv0[12]);
input2 = _mm256_set1_ps(iv0[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv0[7]); 
input1 = _mm256_set1_ps(iv0[13]);
input2 = _mm256_set1_ps(iv0[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv0[8]); 
input1 = _mm256_set1_ps(iv0[14]);
input2 = _mm256_set1_ps(iv0[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
// 下面将第一行的三层参数乘 iv2的东西
input0 = _mm256_set1_ps(iv2[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv2[6]);
input2 = _mm256_set1_ps(iv2[12]);
res00 = input0 * w000;
res01 = input1 * w000;
res02 = input2 * w000;

input0 = _mm256_set1_ps(iv2[1]); 
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
// 更新w000
w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 9 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w001, res02);

input0 = _mm256_set1_ps(iv2[2]); 
input1 = _mm256_set1_ps(iv2[8]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
// 更新w001
w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 10 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w002, res02);
// 更新w002
w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 11 * BLOCK]);
input0 = _mm256_set1_ps(iv2[3]); 
input1 = _mm256_set1_ps(iv2[9]);
input2 = _mm256_set1_ps(iv2[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);

input0 = _mm256_set1_ps(iv2[4]); 
input1 = _mm256_set1_ps(iv2[10]);
input2 = _mm256_set1_ps(iv2[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
// 更新w010
w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 12 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w011, res02);

input0 = _mm256_set1_ps(iv2[5]); 
input1 = _mm256_set1_ps(iv2[11]);
input2 = _mm256_set1_ps(iv2[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
// 更新w011
w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 13 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w012, res02);

input0 = _mm256_set1_ps(iv2[6]); 
input1 = _mm256_set1_ps(iv2[12]);
input2 = _mm256_set1_ps(iv2[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
// 更新w012
w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 14 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w020, res02);

input0 = _mm256_set1_ps(iv2[7]); 
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
// 更新w020
w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 15 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w021, res02);

input0 = _mm256_set1_ps(iv2[8]); 
input1 = _mm256_set1_ps(iv2[14]);
input2 = _mm256_set1_ps(iv2[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
// 更新w021
w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 16 * BLOCK]);
res02 = _mm256_fmadd_ps(input2, w022, res02);
// 更新w022
w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 17 * BLOCK]);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

// 开始将第二行的三层参数load进来
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2]);
input0 = _mm256_set1_ps(iv1[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv1[6]);
input2 = _mm256_set1_ps(iv1[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv1[1]); 
input1 = _mm256_set1_ps(iv1[7]);
input2 = _mm256_set1_ps(iv1[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv1[2]); 
input1 = _mm256_set1_ps(iv1[8]);
input2 = _mm256_set1_ps(iv1[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv1[3]); 
input1 = _mm256_set1_ps(iv1[9]);
input2 = _mm256_set1_ps(iv1[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv1[4]); 
input1 = _mm256_set1_ps(iv1[10]);
input2 = _mm256_set1_ps(iv1[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv1[5]); 
input1 = _mm256_set1_ps(iv1[11]);
input2 = _mm256_set1_ps(iv1[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv1[6]); 
input1 = _mm256_set1_ps(iv1[12]);
input2 = _mm256_set1_ps(iv1[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv1[7]); 
input1 = _mm256_set1_ps(iv1[13]);
input2 = _mm256_set1_ps(iv1[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv1[8]); 
input1 = _mm256_set1_ps(iv1[14]);
input2 = _mm256_set1_ps(iv1[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK]);
input0 = _mm256_set1_ps(iv3[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv3[6]);
input2 = _mm256_set1_ps(iv3[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
// 更新w000
w000 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 18 * BLOCK]);
input0 = _mm256_set1_ps(iv3[1]); 
input1 = _mm256_set1_ps(iv3[7]);
input2 = _mm256_set1_ps(iv3[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
// 更新w001
w001 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 19 * BLOCK]);
input0 = _mm256_set1_ps(iv3[2]); 
input1 = _mm256_set1_ps(iv3[8]);
input2 = _mm256_set1_ps(iv3[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
// 更新w002
w002 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 20 * BLOCK]);
input0 = _mm256_set1_ps(iv3[3]); 
input1 = _mm256_set1_ps(iv3[9]);
input2 = _mm256_set1_ps(iv3[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
// 更新w010
w010 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 21 * BLOCK]);
input0 = _mm256_set1_ps(iv3[4]); 
input1 = _mm256_set1_ps(iv3[10]);
input2 = _mm256_set1_ps(iv3[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
// 更新w011
w011 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 22 * BLOCK]);
input0 = _mm256_set1_ps(iv3[5]); 
input1 = _mm256_set1_ps(iv3[11]);
input2 = _mm256_set1_ps(iv3[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
// 更新w012
w012 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 23 * BLOCK]);
input0 = _mm256_set1_ps(iv3[6]); 
input1 = _mm256_set1_ps(iv3[12]);
input2 = _mm256_set1_ps(iv3[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
// 更新w020
w020 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 24 * BLOCK]);
input0 = _mm256_set1_ps(iv3[7]); 
input1 = _mm256_set1_ps(iv3[13]);
input2 = _mm256_set1_ps(iv3[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);

input0 = _mm256_set1_ps(iv3[8]); 
input1 = _mm256_set1_ps(iv3[14]);
input2 = _mm256_set1_ps(iv3[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
// 更新w021
w021 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 25 * BLOCK]);
// 更新w022
w022 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 26 * BLOCK]);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

// 开始将第三行的三层参数load进来
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2]);
input0 = _mm256_set1_ps(iv2[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv2[6]);
input2 = _mm256_set1_ps(iv2[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv2[1]); 
input1 = _mm256_set1_ps(iv2[7]);
input2 = _mm256_set1_ps(iv2[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv2[2]); 
input1 = _mm256_set1_ps(iv2[8]);
input2 = _mm256_set1_ps(iv2[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv2[3]); 
input1 = _mm256_set1_ps(iv2[9]);
input2 = _mm256_set1_ps(iv2[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv2[4]); 
input1 = _mm256_set1_ps(iv2[10]);
input2 = _mm256_set1_ps(iv2[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv2[5]); 
input1 = _mm256_set1_ps(iv2[11]);
input2 = _mm256_set1_ps(iv2[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv2[6]); 
input1 = _mm256_set1_ps(iv2[12]);
input2 = _mm256_set1_ps(iv2[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv2[7]); 
input1 = _mm256_set1_ps(iv2[13]);
input2 = _mm256_set1_ps(iv2[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv2[8]); 
input1 = _mm256_set1_ps(iv2[14]);
input2 = _mm256_set1_ps(iv2[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK]);
input0 = _mm256_set1_ps(iv4[0]); // 这三个我打算复用呢  
input1 = _mm256_set1_ps(iv4[6]);
input2 = _mm256_set1_ps(iv4[12]);
res00 = _mm256_fmadd_ps(input0, w000, res00);
res01 = _mm256_fmadd_ps(input1, w000, res01);
res02 = _mm256_fmadd_ps(input2, w000, res02);
input0 = _mm256_set1_ps(iv4[1]); 
input1 = _mm256_set1_ps(iv4[7]);
input2 = _mm256_set1_ps(iv4[13]);
res00 = _mm256_fmadd_ps(input0, w001, res00);
res01 = _mm256_fmadd_ps(input1, w001, res01);
res02 = _mm256_fmadd_ps(input2, w001, res02);
input0 = _mm256_set1_ps(iv4[2]); 
input1 = _mm256_set1_ps(iv4[8]);
input2 = _mm256_set1_ps(iv4[14]);
res00 = _mm256_fmadd_ps(input0, w002, res00);
res01 = _mm256_fmadd_ps(input1, w002, res01);
res02 = _mm256_fmadd_ps(input2, w002, res02);
input0 = _mm256_set1_ps(iv4[3]); 
input1 = _mm256_set1_ps(iv4[9]);
input2 = _mm256_set1_ps(iv4[15]);
res00 = _mm256_fmadd_ps(input0, w010, res00);
res01 = _mm256_fmadd_ps(input1, w010, res01);
res02 = _mm256_fmadd_ps(input2, w010, res02);
input0 = _mm256_set1_ps(iv4[4]); 
input1 = _mm256_set1_ps(iv4[10]);
input2 = _mm256_set1_ps(iv4[16]);
res00 = _mm256_fmadd_ps(input0, w011, res00);
res01 = _mm256_fmadd_ps(input1, w011, res01);
res02 = _mm256_fmadd_ps(input2, w011, res02);
input0 = _mm256_set1_ps(iv4[5]); 
input1 = _mm256_set1_ps(iv4[11]);
input2 = _mm256_set1_ps(iv4[17]);
res00 = _mm256_fmadd_ps(input0, w012, res00);
res01 = _mm256_fmadd_ps(input1, w012, res01);
res02 = _mm256_fmadd_ps(input2, w012, res02);
input0 = _mm256_set1_ps(iv4[6]); 
input1 = _mm256_set1_ps(iv4[12]);
input2 = _mm256_set1_ps(iv4[18]);
res00 = _mm256_fmadd_ps(input0, w020, res00);
res01 = _mm256_fmadd_ps(input1, w020, res01);
res02 = _mm256_fmadd_ps(input2, w020, res02);
input0 = _mm256_set1_ps(iv4[7]); 
input1 = _mm256_set1_ps(iv4[13]);
input2 = _mm256_set1_ps(iv4[19]);
res00 = _mm256_fmadd_ps(input0, w021, res00);
res01 = _mm256_fmadd_ps(input1, w021, res01);
res02 = _mm256_fmadd_ps(input2, w021, res02);
input0 = _mm256_set1_ps(iv4[8]); 
input1 = _mm256_set1_ps(iv4[14]);
input2 = _mm256_set1_ps(iv4[20]);
res00 = _mm256_fmadd_ps(input0, w022, res00);
res01 = _mm256_fmadd_ps(input1, w022, res01);
res02 = _mm256_fmadd_ps(input2, w022, res02);

_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

/*第4444种方法结束了*/


          }
        }
    }
    //double convtime2 = (double)clock() / CLOCKS_PER_SEC;
    //std::cout << "这次卷积完成花了" << (convtime2 - convtime1) * 1000 << std::endl;    

/*
    for (int oc_i = 0; oc_i < oc / BLOCK; oc_i++)
    {
      for (int oh_i = 0; oh_i < oh; oh_i += 1)
      {
        for (int ow_i = 0; ow_i < ow; ow_i += 8)
        {
          int start_index = oc_i * oh * ow * BLOCK + oh_i * ow * BLOCK + ow_i * BLOCK;//trans_out;
          __m256 row0 = _mm256_loadu_ps(&trans_out[start_index]);
          __m256 row1 = _mm256_loadu_ps(&trans_out[start_index + 1 * 8]);
          __m256 row2 = _mm256_loadu_ps(&trans_out[start_index + 2 * 8]);
          __m256 row3 = _mm256_loadu_ps(&trans_out[start_index + 3 * 8]);
          __m256 row4 = _mm256_loadu_ps(&trans_out[start_index + 4 * 8]);
          __m256 row5 = _mm256_loadu_ps(&trans_out[start_index + 5 * 8]);
          __m256 row6 = _mm256_loadu_ps(&trans_out[start_index + 6 * 8]);
          __m256 row7 = _mm256_loadu_ps(&trans_out[start_index + 7 * 8]);
          my_transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
          int dst_index = oc_i * BLOCK * oh * ow + oh_i * ow + ow_i;
          // 这里需要加bias和激活函数
          _mm256_storeu_ps(&o_data[dst_index + 0 * oh * ow], row0);
          _mm256_storeu_ps(&o_data[dst_index + 1 * oh * ow], row1);
          _mm256_storeu_ps(&o_data[dst_index + 2 * oh * ow], row2);
          _mm256_storeu_ps(&o_data[dst_index + 3 * oh * ow], row3);
          _mm256_storeu_ps(&o_data[dst_index + 4 * oh * ow], row4);
          _mm256_storeu_ps(&o_data[dst_index + 5 * oh * ow], row5);
          _mm256_storeu_ps(&o_data[dst_index + 6 * oh * ow], row6);
          _mm256_storeu_ps(&o_data[dst_index + 7 * oh * ow], row7);
        }
      }
    }
*/

 double trans_out_time1 = (double)clock() / CLOCKS_PER_SEC;
    for (int oc_i = 0; oc_i < oc; oc_i ++)
    {
      for (int oh_i = 0; oh_i < oh; oh_i ++)
      {
        for (int ow_i = 0; ow_i < ow; ow_i ++)
        {
          int start_o1 = bs_i * oc * oh * ow + oc_i * oh * ow + oh_i * ow + ow_i;
          int start_t1 = oc_i / BLOCK * oh * ow * BLOCK + oh_i * ow * BLOCK + ow_i * BLOCK + oc_i % BLOCK;
          o_data [start_o1] = 
                    trans_out[start_t1];
        }
      }
    }
  double trans_out_time2 = (double)clock() / CLOCKS_PER_SEC;
  std::cout << "转换输出的时间" <<(trans_out_time2 -trans_out_time1) * 1000 << "ms"<< std::endl;

/*
    for (int oc_i = 0; oc_i < oc / BLOCK; oc_i++)
    {
      for (int oh_i = 0; oh_i < oh; oh_i += 1)
      {
        for (int ow_i = 0; ow_i < ow; ow_i += 8)
        {
          int start_index = oc_i * oh * ow * BLOCK + oh_i * ow * BLOCK + ow_i * BLOCK;//trans_out;
          __m256 row0 = _mm256_loadu_ps(&trans_out[start_index]);
          __m256 row1 = _mm256_loadu_ps(&trans_out[start_index + 1 * 8]);
          __m256 row2 = _mm256_loadu_ps(&trans_out[start_index + 2 * 8]);
          __m256 row3 = _mm256_loadu_ps(&trans_out[start_index + 3 * 8]);
          __m256 row4 = _mm256_loadu_ps(&trans_out[start_index + 4 * 8]);
          __m256 row5 = _mm256_loadu_ps(&trans_out[start_index + 5 * 8]);
          __m256 row6 = _mm256_loadu_ps(&trans_out[start_index + 6 * 8]);
          __m256 row7 = _mm256_loadu_ps(&trans_out[start_index + 7 * 8]);
          my_transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
          int dst_index = oc_i * BLOCK * oh * ow + oh_i * ow + ow_i;
          // 这里需要加bias和激活函数
          _mm256_storeu_ps(&o_data[dst_index + 0 * oh * ow], row0);
          _mm256_storeu_ps(&o_data[dst_index + 1 * oh * ow], row1);
          _mm256_storeu_ps(&o_data[dst_index + 2 * oh * ow], row2);
          _mm256_storeu_ps(&o_data[dst_index + 3 * oh * ow], row3);
          _mm256_storeu_ps(&o_data[dst_index + 4 * oh * ow], row4);
          _mm256_storeu_ps(&o_data[dst_index + 5 * oh * ow], row5);
          _mm256_storeu_ps(&o_data[dst_index + 6 * oh * ow], row6);
          _mm256_storeu_ps(&o_data[dst_index + 7 * oh * ow], row7);
        }
      }
    }
*/
    time2 = (double)clock() / CLOCKS_PER_SEC;
    std::cout << "全部卷积的时间" <<(time2 -time1) * 1000 << "ms"<< std::endl;
  }
}

void conv_direct_3x3s2(const float* i_data,
                            const float* trans_weight,
                            float* trans_out,  // holds the intermediate output result  
                            int bs,
                            int ic,
                            int ih,
                            int iw,
                            int oc,
                            int oc_expand,
                            float* o_data,
                            int oh, int ow, int ph, int pw,
                            const float* bias, lite_api::ActivationType active_type)
{
  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int strideh = 2;
  constexpr int stridew = 2;

#ifdef __AVX__
  constexpr int BLOCK  = 8;
  constexpr int window_h = 5; //the sliding window is 5x7 and can obtain 2x3 results！ for AVX
  constexpr int window_w = 7;

#else
  constexpr int BLOCK  = 4;
  constexpr int window_h = 5;
  constexpr int window_w = 7;
#endif

  int wc = ic;
  for (int bs_i = 0; bs_i < bs; bs_i ++) // fetch bs_i th input feature map
  {
    memset(trans_out, 0, sizeof(float)  * oc_expand / BLOCK * oh * ow * BLOCK);
    for (int ic_i = 0; ic_i < wc; ic_i ++) // fetch the ic_i th channel in this input feature map
    {
      for (int group_i = 0; group_i < oc_expand / BLOCK; group_i ++)// fetch group_i th group kernel,there are BLOCK kernels in it. Naturally, we have to deal with its ic_i channel
      {
        // Now, we need compute the conv of one planar feature map and BLOCK  planar kernel
        int input_start_index = bs_i * ic * ih * iw + ic_i * ih * iw; // the  planar feature map's starting address
        int kernel_start_index = group_i * wc * wh * ww * BLOCK + ic_i * wh * ww * BLOCK; // the first kernel's address in this BLOCK
        int output_start_index = group_i * oh * ow * BLOCK;
        
        int new_ih; // The maximum value of the upper left corner of the sliding window in h dimension    
        int new_iw; 
        int new_ih_start;
        int new_iw_start;
        if (ph == 0 && pw == 0)
        {
            new_iw = (iw - window_w) / 6 * 6; // 6 is the stride_w of sliding window
            new_ih = (ih - window_h) / 4 * 4; 
            new_ih_start = 0;
            new_iw_start = 0;
        }
        else if(ph == 1 && pw == 1)
        {
            new_iw = (iw - window_w - 1) / 6 * 6 + 1; 
            new_ih = (ih - window_h - 1) / 4 * 4 + 1;
            new_ih_start = 1;
            new_iw_start = 1;
        }

        int o_left = (new_iw_start + pw) / 2;  //[0,o_left) in output map needs Special treatment
        int o_right = (new_iw + pw) / 2 + 3;   // [o_right, ow) in output map needs Special treatment
        int o_upper = (new_ih_start + ph) / 2; //[0,o_upper) same as above
        int o_down = (new_ih + ph) / 2 + 2;    // [o_down, oh) same as above 

        for (int oh_i = 0; oh_i < o_upper; oh_i ++)
        {
            for (int ow_i = 0; ow_i < ow; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;

                // oh_i and ow_i is the index of the output. 
                // Next, calculate the index of their corresponding input.
                // These two are in the upper left corner of the corresponding input!
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw; 
                
                // Let's start the convolution of 3x3!
#ifdef __AVX__
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
#else
                __m128 res = _mm_loadu_ps(&trans_out[output_index]);
#endif
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
#else
                    __m128 input = _mm_set1_ps(i_data[input_index]);
                    __m128 w = _mm_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm_fmadd_ps(input, w, res); 
#endif
                }
#ifdef __AVX__
                _mm256_storeu_ps(&trans_out[output_index], res); 
#else
                _mm_storeu_ps(&trans_out[output_index], res);
#endif        
            }
        }

        for (int oh_i = o_down; oh_i < oh; oh_i ++)
        {
            for (int ow_i = 0; ow_i < ow; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;
#ifdef __AVX__
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
#else
                __m128 res = _mm_loadu_ps(&trans_out[output_index]);
#endif              
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
#else
                    __m128 input = _mm_set1_ps(i_data[input_index]);
                    __m128 w = _mm_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm_fmadd_ps(input, w, res); 
#endif
                }
#ifdef __AVX__
                   _mm256_storeu_ps(&trans_out[output_index], res);
#else
                   _mm_storeu_ps(&trans_out[output_index], res);
#endif

            }
        }

        for (int oh_i = 0; oh_i < oh; oh_i ++)
        {
            if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
                continue;
            for (int ow_i = 0; ow_i < o_left; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;
#ifdef __AVX__
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
#else
                __m128 res = _mm_loadu_ps(&trans_out[output_index]);
#endif
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
#else
                    __m128 input = _mm_set1_ps(i_data[input_index]);
                    __m128 w = _mm_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm_fmadd_ps(input, w, res); 
#endif
                }
#ifdef __AVX__
                   _mm256_storeu_ps(&trans_out[output_index], res);
#else
                   _mm_storeu_ps(&trans_out[output_index], res);
#endif
            }
        }

        for (int oh_i = 0; oh_i < oh; oh_i ++)
        {
            if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
                continue;
            for (int ow_i = o_right; ow_i < ow; ow_i ++)
            {
                int output_index = output_start_index +  oh_i * ow * BLOCK + ow_i * BLOCK;
                int ih_i = oh_i * strideh - ph;
                int iw_i = ow_i * stridew - pw;
#ifdef __AVX__
                __m256 res = _mm256_loadu_ps(&trans_out[output_index]);
#else
                __m128 res = _mm_loadu_ps(&trans_out[output_index]);
#endif
                for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    int new_ih_i = ih_i + i;
                    int new_iw_i = iw_i + j;
                    if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 || new_iw_i >= iw) continue;
                    int input_index = input_start_index + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                    __m256 input = _mm256_set1_ps(i_data[input_index]);
                    __m256 w = _mm256_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm256_fmadd_ps(input, w, res); 
#else
                    __m128 input = _mm_set1_ps(i_data[input_index]);
                    __m128 w = _mm_loadu_ps(&trans_weight[kernel_start_index + (i * 3 + j) * BLOCK]);
                    res = _mm_fmadd_ps(input, w, res); 
#endif
                }
#ifdef __AVX__
                 _mm256_storeu_ps(&trans_out[output_index], res);
#else
                _mm_storeu_ps(&trans_out[output_index], res);
#endif
            }
        }

        // So far, we have dealt with the special boundary, 
        // and now we begin to deal with the general situation
        for (int ih_i = new_ih_start; ih_i <= new_ih; ih_i += 4)
        {
          for (int iw_i = new_iw_start; iw_i <= new_iw; iw_i += 6)
          {
            int output_index = output_start_index + (ih_i + ph) / 2 * ow * BLOCK + (iw_i + pw) / 2 * BLOCK;


// The following is the starting address of each line of the sliding window
const  float* iv0 = i_data + input_start_index + ih_i * iw + 0 * iw + iw_i;
const  float* iv1 = iv0 + iw;
const  float* iv2 = iv0 + 2 * iw;
const  float* iv3 = iv0 + 3 * iw;
const  float* iv4 = iv0 + 4 * iw;

#ifdef __AVX__

            // Take out 9 weight values to the register
            __m256 w00 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 0 * BLOCK]);  
            __m256 w01 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 1 * BLOCK]);
            __m256 w02 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 2 * BLOCK]);
            __m256 w10 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 3 * BLOCK]);
            __m256 w11 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 4 * BLOCK]);
            __m256 w12 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 5 * BLOCK]);  
            __m256 w20 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 6 * BLOCK]);
            __m256 w21 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 7 * BLOCK]);
            __m256 w22 = _mm256_loadu_ps(&trans_weight[kernel_start_index + 8 * BLOCK]);

// Sliding windows can produce 2x3 results, but I now create three __m256
// to  represent the outputs  in the first line  
__m256 res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0]);
__m256 res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1]);
__m256 res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2]);

// I have used 12 registers, and there are 4 left!
// Next, I will use three to hold input data to generate outputs !
// iv0: 0 1 2 3 4 5 6 
// 0,1,2 is Responsible for res00
// 2,3,4 is Responsible for res01
// 4,5,6 is Responsible for res01
__m256 input0 = _mm256_set1_ps(iv0[0]);  
__m256 input2 = _mm256_set1_ps(iv0[2]);  // This needs to be retained because of reuse
__m256 input4 = _mm256_set1_ps(iv0[4]);  // This needs to be retained because of reuse
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
input2 = _mm256_set1_ps(iv1[2]);  // This needs to be retained because of reuse
input4 = _mm256_set1_ps(iv1[4]);  // This needs to be retained because of reuse
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
//The first, second and third results have been calculated above. Store them back
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);

// res00,res01,res02 need to be updated representing the outputs in the second line 
// iv2: 0 1 2 3 4 5 6 
// 0,1,2 is Responsible for res00
// 2,3,4 is Responsible for res01
// 4,5,6 is Responsible for res01
res00 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK]);
res01 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK]);
res02 = _mm256_loadu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK]);
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
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm256_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

#else

            // Take out 9 weight values to the register
            __m128 w00 = _mm_loadu_ps(&trans_weight[kernel_start_index + 0 * BLOCK]);  
            __m128 w01 = _mm_loadu_ps(&trans_weight[kernel_start_index + 1 * BLOCK]);
            __m128 w02 = _mm_loadu_ps(&trans_weight[kernel_start_index + 2 * BLOCK]);
            __m128 w10 = _mm_loadu_ps(&trans_weight[kernel_start_index + 3 * BLOCK]);
            __m128 w11 = _mm_loadu_ps(&trans_weight[kernel_start_index + 4 * BLOCK]);
            __m128 w12 = _mm_loadu_ps(&trans_weight[kernel_start_index + 5 * BLOCK]);  
            __m128 w20 = _mm_loadu_ps(&trans_weight[kernel_start_index + 6 * BLOCK]);
            __m128 w21 = _mm_loadu_ps(&trans_weight[kernel_start_index + 7 * BLOCK]);
            __m128 w22 = _mm_loadu_ps(&trans_weight[kernel_start_index + 8 * BLOCK]);

// Sliding windows can produce 2x3 results, but I now create three __m256
// to  represent the outputs  in the first line  
__m128 res00 = _mm_loadu_ps(&trans_out[output_index + BLOCK * 0]);
__m128 res01 = _mm_loadu_ps(&trans_out[output_index + BLOCK * 1]);
__m128 res02 = _mm_loadu_ps(&trans_out[output_index + BLOCK * 2]);

// I have used 12 registers, and there are 4 left!
// Next, I will use three to hold input data to generate outputs !
// iv0: 0 1 2 3 4 5 6 
// 0,1,2 is Responsible for res00
// 2,3,4 is Responsible for res01
// 4,5,6 is Responsible for res01
__m128 input0 = _mm_set1_ps(iv0[0]);  
__m128 input2 = _mm_set1_ps(iv0[2]);  // This needs to be retained because of reuse
__m128 input4 = _mm_set1_ps(iv0[4]);  // This needs to be retained because of reuse
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
input2 = _mm_set1_ps(iv1[2]);  // This needs to be retained because of reuse
input4 = _mm_set1_ps(iv1[4]);  // This needs to be retained because of reuse
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
//The first, second and third results have been calculated above. Store them back
_mm_storeu_ps(&trans_out[output_index + BLOCK * 0], res00);
_mm_storeu_ps(&trans_out[output_index + BLOCK * 1], res01);
_mm_storeu_ps(&trans_out[output_index + BLOCK * 2], res02);

// res00,res01,res02 need to be updated representing the outputs in the second line 
// iv2: 0 1 2 3 4 5 6 
// 0,1,2 is Responsible for res00
// 2,3,4 is Responsible for res01
// 4,5,6 is Responsible for res01
res00 = _mm_loadu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK]);
res01 = _mm_loadu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK]);
res02 = _mm_loadu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK]);
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
_mm_storeu_ps(&trans_out[output_index + BLOCK * 0 + ow * BLOCK], res00);
_mm_storeu_ps(&trans_out[output_index + BLOCK * 1 + ow * BLOCK], res01);
_mm_storeu_ps(&trans_out[output_index + BLOCK * 2 + ow * BLOCK], res02);

#endif
          }
        }
      }
    }

    // convert trans_out(HWC) to o_data(CHW)!
    for (int oc_i = 0; oc_i < oc / BLOCK; oc_i++)
    {
      for (int oh_i = 0; oh_i < oh; oh_i += 1)
      {
        for (int ow_i = 0; ow_i < ow / BLOCK * BLOCK; ow_i += BLOCK)
        {
          //trans_out's start_index, we need fetch 8x8 element; 
          int start_index = oc_i * oh * ow * BLOCK + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__
          __m256 row0 = _mm256_loadu_ps(&trans_out[start_index + 0 * BLOCK]);
          __m256 row1 = _mm256_loadu_ps(&trans_out[start_index + 1 * BLOCK]);
          __m256 row2 = _mm256_loadu_ps(&trans_out[start_index + 2 * BLOCK]);
          __m256 row3 = _mm256_loadu_ps(&trans_out[start_index + 3 * BLOCK]);
          __m256 row4 = _mm256_loadu_ps(&trans_out[start_index + 4 * BLOCK]);
          __m256 row5 = _mm256_loadu_ps(&trans_out[start_index + 5 * BLOCK]);
          __m256 row6 = _mm256_loadu_ps(&trans_out[start_index + 6 * BLOCK]);
          __m256 row7 = _mm256_loadu_ps(&trans_out[start_index + 7 * BLOCK]);
          conv_direct_transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
#else

          __m128 row0 = _mm_loadu_ps(&trans_out[start_index + 0 * BLOCK]);
          __m128 row1 = _mm_loadu_ps(&trans_out[start_index + 1 * BLOCK]);
          __m128 row2 = _mm_loadu_ps(&trans_out[start_index + 2 * BLOCK]);
          __m128 row3 = _mm_loadu_ps(&trans_out[start_index + 3 * BLOCK]);
          conv_direct_transpose4_ps(row0, row1, row2, row3);
#endif

          int dst_index = oc_i * BLOCK * oh * ow + oh_i * ow + ow_i;
          if (bias != nullptr)
          {
#ifdef __AVX__
            row0 = _mm256_add_ps(row0, _mm256_set1_ps(bias[oc_i * BLOCK + 0]));
            row1 = _mm256_add_ps(row1, _mm256_set1_ps(bias[oc_i * BLOCK + 1]));
            row2 = _mm256_add_ps(row2, _mm256_set1_ps(bias[oc_i * BLOCK + 2]));
            row3 = _mm256_add_ps(row3, _mm256_set1_ps(bias[oc_i * BLOCK + 3]));
            row4 = _mm256_add_ps(row4, _mm256_set1_ps(bias[oc_i * BLOCK + 4]));
            row5 = _mm256_add_ps(row5, _mm256_set1_ps(bias[oc_i * BLOCK + 5]));
            row6 = _mm256_add_ps(row6, _mm256_set1_ps(bias[oc_i * BLOCK + 6]));
            row7 = _mm256_add_ps(row7, _mm256_set1_ps(bias[oc_i * BLOCK + 7]));
#else
            row0 = _mm_add_ps(row0, _mm_set1_ps(bias[oc_i * BLOCK + 0]));
            row1 = _mm_add_ps(row1, _mm_set1_ps(bias[oc_i * BLOCK + 1]));
            row2 = _mm_add_ps(row2, _mm_set1_ps(bias[oc_i * BLOCK + 2]));
            row3 = _mm_add_ps(row3, _mm_set1_ps(bias[oc_i * BLOCK + 3]));
#endif
          }
          if (active_type == lite_api::ActivationType::kRelu)
          {
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
          }
          else if (active_type == lite_api::ActivationType::kRelu6)
          {
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
          }
          else if (active_type == lite_api::ActivationType::kIndentity)
          {
          }
          else
          {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }
#ifdef __AVX__
          _mm256_storeu_ps(&o_data[dst_index + 0 * oh * ow], row0);
          _mm256_storeu_ps(&o_data[dst_index + 1 * oh * ow], row1);
          _mm256_storeu_ps(&o_data[dst_index + 2 * oh * ow], row2);
          _mm256_storeu_ps(&o_data[dst_index + 3 * oh * ow], row3);
          _mm256_storeu_ps(&o_data[dst_index + 4 * oh * ow], row4);
          _mm256_storeu_ps(&o_data[dst_index + 5 * oh * ow], row5);
          _mm256_storeu_ps(&o_data[dst_index + 6 * oh * ow], row6);
          _mm256_storeu_ps(&o_data[dst_index + 7 * oh * ow], row7);
#else
          _mm_storeu_ps(&o_data[dst_index + 0 * oh * ow], row0);
          _mm_storeu_ps(&o_data[dst_index + 1 * oh * ow], row1);
          _mm_storeu_ps(&o_data[dst_index + 2 * oh * ow], row2);
          _mm_storeu_ps(&o_data[dst_index + 3 * oh * ow], row3);
#endif
        }
        for (int ow_i = ow / BLOCK * BLOCK; ow_i < ow; ow_i ++)
        {
          // trans_out
          int start_index = oc_i * oh * ow * BLOCK + oh_i * ow * BLOCK + ow_i * BLOCK;
          int dst_index = oc_i * BLOCK * oh * ow + oh_i * ow + ow_i;
#ifdef __AVX__
          __m256 row = _mm256_loadu_ps(&trans_out[start_index]);
#else
          __m128 row = _mm_loadu_ps(&trans_out[start_index]);
#endif
          if (bias != nullptr)
          {
#ifdef __AVX__
            row = _mm256_add_ps(row, _mm256_loadu_ps(&bias[oc_i * BLOCK]));
#else
            row = _mm_add_ps(row, _mm_loadu_ps(&bias[oc_i * BLOCK]));
#endif
          }
          if (active_type == lite_api::ActivationType::kRelu)
          {
#ifdef __AVX__
          row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
#else
         row = _mm_max_ps(row, _mm_set1_ps(0.f));
#endif
          }
          else if (active_type == lite_api::ActivationType::kRelu6)
          {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
            row = _mm256_min_ps(row, _mm256_set1_ps(6.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
            row = _mm_min_ps(row, _mm_set1_ps(6.f));
#endif
          }
          else if (active_type == lite_api::ActivationType::kIndentity)
          {
          }
          else
          {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }
#ifdef __AVX__
          o_data[dst_index + 0 * oh * ow] = row[0];
          o_data[dst_index + 1 * oh * ow] = row[1];
          o_data[dst_index + 2 * oh * ow] = row[2];
          o_data[dst_index + 3 * oh * ow] = row[3];
          o_data[dst_index + 4 * oh * ow] = row[4];
          o_data[dst_index + 5 * oh * ow] = row[5];
          o_data[dst_index + 6 * oh * ow] = row[6];
          o_data[dst_index + 7 * oh * ow] = row[7];
#else
          o_data[dst_index + 0 * oh * ow] = row[0];
          o_data[dst_index + 1 * oh * ow] = row[1];
          o_data[dst_index + 2 * oh * ow] = row[2];
          o_data[dst_index + 3 * oh * ow] = row[3];
#endif
        }
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
