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

#include "lite/backends/arm/math/im2sequence.h"
#include <arm_neon.h>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void im2sequence(const float* input,
                 const int input_c,
                 const int input_h,
                 const int input_w,
                 const int kernel_h,
                 const int kernel_w,
                 const int pad_top,
                 const int pad_bottom,
                 const int pad_left,
                 const int pad_right,
                 const int stride_h,
                 const int stride_w,
                 const int out_h,
                 const int out_w,
                 float* out,
                 Context<TARGET(kARM)>* ctx) {
  int window_size = kernel_h * kernel_w;
  int out_rows = out_h * out_w;
  int out_cols = input_c * window_size;
  int H_pad = input_h + pad_top + pad_bottom;
  int W_pad = input_w + pad_left + pad_right;
  for (int h_id = 0; h_id < out_h; h_id++) {
    for (int w_id = 0; w_id < out_w; w_id++) {
      // consider dilation.
      int start_h = h_id * stride_h - pad_top;
      int start_w = w_id * stride_w - pad_left;
      for (int c_id = 0; c_id < input_c; c_id++) {
        for (int k_h_id = 0; k_h_id < kernel_h; k_h_id++) {
          int in_h_id = start_h + k_h_id;
          bool exceed_flag = (in_h_id < 0) || (in_h_id >= H_pad);
          int out_start_id =
              (h_id * out_w + w_id) * out_cols + c_id * window_size;
          for (int k_w_id = 0; k_w_id < kernel_w; k_w_id++) {
            int in_w_id = start_w + k_w_id;
            exceed_flag = exceed_flag || (in_w_id < 0) || (in_w_id >= W_pad);
            int input_id = (c_id * input_h + in_h_id) * input_w + in_w_id;
            int out_id = out_start_id + k_h_id * kernel_w + k_w_id;
            out[out_id] = exceed_flag ? 0.f : input[input_id];
          }
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
