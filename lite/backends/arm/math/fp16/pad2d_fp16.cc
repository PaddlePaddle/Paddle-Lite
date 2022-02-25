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

#include "lite/backends/arm/math/fp16/pad2d_fp16.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void pad_constant_fp16(const float16_t* din,
                       float16_t* dout,
                       int n,
                       int c,
                       int h,
                       int w,
                       const int pad_top,
                       const int pad_bottom,
                       const int pad_left,
                       const int pad_right,
                       const float16_t pad_value) {
  int h_in = h - pad_top - pad_bottom;
  int w_in = w - pad_left - pad_right;
  int spatial_size_out = w * h;
  int spatial_size_in = h_in * w_in;
  int num = n * c;
  int top_loop = (w * pad_top) >> 3;
  int top_loop_remain = (w * pad_top) & 7;
  int left_loop = pad_left >> 2;
  int right_loop = pad_right >> 2;
  int left_loop_remain = pad_left & 3;
  int right_loop_remain = pad_right & 3;
  float16x8_t vpad_value = vdupq_n_f16(pad_value);
  float16x4_t vpad_value_low = vdup_n_f16(pad_value);
  int bottom_loop = (w * pad_bottom) >> 3;
  int bottom_loop_remain = (w * pad_bottom) & 7;

  LITE_PARALLEL_BEGIN(s, tid, num) {
    const float16_t* din_s = din + s * spatial_size_in;
    float16_t* dout_s = dout + s * spatial_size_out;
    // process top
    for (int i = 0; i < top_loop; ++i) {
      vst1q_f16(dout_s, vpad_value);
      dout_s += 8;
    }
    for (int i = 0; i < top_loop_remain; ++i) {
      *dout_s++ = pad_value;
    }
    // process med
    for (int j = 0; j < h_in; ++j) {
      for (int i = 0; i < left_loop; ++i) {
        vst1_f16(dout_s, vpad_value_low);
        dout_s += 4;
      }
      for (int i = 0; i < left_loop_remain; ++i) {
        *dout_s++ = pad_value;
      }
      memcpy(dout_s, din_s, sizeof(float16_t) * w_in);
      dout_s += w_in;
      din_s += w_in;
      for (int i = 0; i < right_loop; ++i) {
        vst1_f16(dout_s, vpad_value_low);
        dout_s += 4;
      }
      for (int i = 0; i < right_loop_remain; ++i) {
        *dout_s++ = pad_value;
      }
    }

    // process bottom
    for (int i = 0; i < bottom_loop; ++i) {
      vst1q_f16(dout_s, vpad_value);
      dout_s += 8;
    }
    for (int i = 0; i < bottom_loop_remain; ++i) {
      *dout_s++ = pad_value;
    }
  }
  LITE_PARALLEL_END()
}

void pad_edge_fp16(const float16_t* din,
                   float16_t* dout,
                   int n,
                   int c,
                   int h,
                   int w,
                   const int pad_top,
                   const int pad_bottom,
                   const int pad_left,
                   const int pad_right,
                   const float16_t pad_value) {
  int h_in = h - pad_top - pad_bottom;
  int w_in = w - pad_left - pad_right;
  int spatial_size_out = w * h;
  int spatial_size_in = h_in * w_in;
  int num = n * c;
  int med_loop = w_in >> 5;
  int left_loop = pad_left >> 2;
  int right_loop = pad_right >> 2;
  int med_loop_remain = w_in & 31;
  int left_loop_remain = pad_left & 3;
  int right_loop_remain = pad_right & 3;
  int med_rem_cnt = med_loop_remain >> 2;
  int med_rem_rem = med_loop_remain & 3;

  LITE_PARALLEL_BEGIN(s, tid, num) {
    const float16_t* din_s = din + s * spatial_size_in;
    float16_t* dout_s = dout + s * spatial_size_out;
    float16_t* dout_med = dout_s + w * pad_top;
    for (int j = 0; j < h_in; ++j) {
      float16_t edge_val = din_s[0];
      float16x4_t vedge = vdup_n_f16(edge_val);
      for (int i = 0; i < left_loop; ++i) {
        vst1_f16(dout_med, vedge);
        dout_med += 4;
      }
      for (int i = 0; i < left_loop_remain; ++i) {
        *dout_med++ = edge_val;
      }
      for (int i = 0; i < med_loop; ++i) {
        float16x8_t val0 = vld1q_f16(din_s);
        float16x8_t val1 = vld1q_f16(din_s + 8);
        float16x8_t val2 = vld1q_f16(din_s + 16);
        float16x8_t val3 = vld1q_f16(din_s + 24);
        din_s += 32;
        vst1q_f16(dout_med, val0);
        vst1q_f16(dout_med + 8, val1);
        vst1q_f16(dout_med + 16, val2);
        vst1q_f16(dout_med + 24, val3);
        dout_med += 32;
      }
      for (int i = 0; i < med_rem_cnt; ++i) {
        float16x4_t val0 = vld1_f16(din_s);
        din_s += 4;
        vst1_f16(dout_med, val0);
        dout_med += 4;
      }
      for (int i = 0; i < med_rem_rem; ++i) {
        *dout_med++ = *din_s++;
      }
      edge_val = din_s[-1];
      vedge = vdup_n_f16(edge_val);
      for (int i = 0; i < right_loop; ++i) {
        vst1_f16(dout_med, vedge);
        dout_med += 4;
      }
      for (int i = 0; i < right_loop_remain; ++i) {
        *dout_med++ = edge_val;
      }
    }

    // process bottom
    float16_t* dout_bottom = dout_med;
    for (int i = 0; i < pad_bottom; ++i) {
      memcpy(dout_bottom,
             dout_s + w * (pad_top + h_in - 1),
             w * sizeof(float16_t));
      dout_bottom += w;
    }

    // process top
    float16_t* dout_top = dout_s;
    for (int i = 0; i < pad_top; ++i) {
      memcpy(dout_top, dout_s + w * pad_top, w * sizeof(float16_t));
      dout_top += w;
    }
  }
  LITE_PARALLEL_END()
}

void pad_reflect_fp16(const float16_t* din,
                      float16_t* dout,
                      int n,
                      int c,
                      int h,
                      int w,
                      const int pad_top,
                      const int pad_bottom,
                      const int pad_left,
                      const int pad_right,
                      const float16_t pad_value) {
  int h_in = h - pad_top - pad_bottom;
  int w_in = w - pad_left - pad_right;
  int spatial_size_out = w * h;
  int spatial_size_in = h_in * w_in;
  int num = n * c;
  int med_loop = w_in >> 5;
  int left_loop = pad_left >> 2;
  int right_loop = pad_right >> 2;
  int med_loop_remain = w_in & 31;
  int left_loop_remain = pad_left & 3;
  int right_loop_remain = pad_right & 3;
  int med_rem_cnt = med_loop_remain >> 2;
  int med_rem_rem = med_loop_remain & 3;

  LITE_PARALLEL_BEGIN(s, tid, num) {
    const float16_t* din_s = din + s * spatial_size_in;
    float16_t* dout_s = dout + s * spatial_size_out;

    // process med
    float16_t* dout_med = dout_s + w * pad_top;
    for (int j = 0; j < h_in; ++j) {
      for (int i = 0; i < left_loop; ++i) {
        float16x4_t val =
            vld1_f16(din_s + left_loop_remain + ((left_loop - i - 1) << 2) + 1);
        val = vrev64_f16(val);
        vst1_f16(dout_med, val);
        dout_med += 4;
      }
      for (int i = 0; i < left_loop_remain; ++i) {
        *dout_med++ = *(din_s + left_loop_remain - i);
      }
      for (int i = 0; i < med_loop; ++i) {
        float16x8_t val0 = vld1q_f16(din_s);
        float16x8_t val1 = vld1q_f16(din_s + 8);
        float16x8_t val2 = vld1q_f16(din_s + 16);
        float16x8_t val3 = vld1q_f16(din_s + 24);
        din_s += 32;
        vst1q_f16(dout_med, val0);
        vst1q_f16(dout_med + 8, val1);
        vst1q_f16(dout_med + 16, val2);
        vst1q_f16(dout_med + 24, val3);
        dout_med += 32;
      }
      for (int i = 0; i < med_rem_cnt; ++i) {
        float16x4_t val0 = vld1_f16(din_s);
        din_s += 4;
        vst1_f16(dout_med, val0);
        dout_med += 4;
      }
      for (int i = 0; i < med_rem_rem; ++i) {
        *dout_med++ = *din_s++;
      }
      for (int i = 0; i < right_loop; ++i) {
        float16x4_t val = vld1_f16(din_s - ((i + 1) << 2) - 1);
        val = vrev64_f16(val);  // different from vrev64_f32
        vst1_f16(dout_med, val);
        dout_med += 4;
      }
      const float16_t* remain = din_s - (right_loop << 2) - 2;
      for (int i = 0; i < right_loop_remain; ++i) {
        *dout_med++ = *remain--;
      }
    }
    // process bottom
    float16_t* dout_bottom = dout_med;
    float16_t* dout_bottom_reflect = dout_med - (w << 1);
    for (int i = 0; i < pad_bottom; ++i) {
      memcpy(dout_bottom, dout_bottom_reflect, w * sizeof(float16_t));
      dout_bottom += w;
      dout_bottom_reflect -= w;
    }

    // process top
    float16_t* dout_top = dout_s;
    float16_t* dout_top_reflect = dout_s + w * (pad_top << 1);
    for (int i = 0; i < pad_top; ++i) {
      memcpy(dout_top, dout_top_reflect, w * sizeof(float16_t));
      dout_top += w;
      dout_top_reflect -= w;
    }
  }
  LITE_PARALLEL_END()
}

void pad2d_func_fp16(const lite::Tensor* input,
                     lite::Tensor* output,
                     int mode,
                     std::vector<int> pad_h,
                     std::vector<int> pad_w,
                     float16_t pad_value) {
  float16_t* dout = output->mutable_data<float16_t>();
  const float16_t* din = input->data<float16_t>();

  auto output_dims = output->dims();
  // Layout = NCHW
  int batch = output_dims[0];
  int ch_out = output_dims[1];
  int oh = output_dims[2];
  int ow = output_dims[3];
/////////////////////////////
/*     _mode是PadMode
       typedef enum{
           PAD_CONSTANT = 0,
           PAD_EDGE = 1,
           PAD_REFLECT = 2,
       } PadMode;   */
/////////////////////////
#define PAD2D_PARAM                                                         \
  din, dout, batch, ch_out, oh, ow, pad_h[0], pad_h[1], pad_w[0], pad_w[1], \
      pad_value
  if (mode == 0) {
    pad_constant_fp16(PAD2D_PARAM);
  } else if (mode == 1) {
    pad_reflect_fp16(PAD2D_PARAM);
  } else if (mode == 2) {
    pad_edge_fp16(PAD2D_PARAM);
  } else {
    LOG(ERROR) << "ERROR: unknown pad mode: " << mode;
  }
#undef PAD2D_PARAM
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
