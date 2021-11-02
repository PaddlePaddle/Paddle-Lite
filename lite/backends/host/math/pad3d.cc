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

#include "lite/backends/host/math/pad3d.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

void pad_ncdhw_constant(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back,
                        const float pad_value) {
  int num = n * c;
  int size_in_hw = in_h * in_w;
  int size_out_hw = out_h * out_w;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, num) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int d = -pad_front; d < in_d + pad_back; d++) {
      if (d < 0 || d >= in_d) {
        memset(dout_num, pad_value, sizeof(float) * size_out_hw);
        dout_num += size_out_hw;
      } else {
        for (int h = -pad_top; h < in_h + pad_bottom; h++) {
          if (h < 0 || h >= in_h) {
            memset(dout_num, pad_value, sizeof(float) * out_w);
            dout_num += out_w;
          } else {
            if (pad_left) {
              memset(dout_num, pad_value, sizeof(float) * pad_left);
              dout_num += pad_left;
            }
            memcpy(dout_num, din_num, sizeof(float) * in_w);
            dout_num += in_w;
            din_num += in_w;
            if (pad_right) {
              memset(dout_num, pad_value, sizeof(float) * pad_right);
              dout_num += pad_right;
            }
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ndhwc_constant(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back,
                        const float pad_value) {
  int in_wc = in_w * c;
  int out_wc = out_w * c;
  int size_in_hw = in_h * in_wc;
  int size_out_hw = out_h * out_wc;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  int pad_leftc = pad_left * c;
  int pad_rightc = pad_right * c;
  LITE_PARALLEL_BEGIN(i, tid, n) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int d = -pad_front; d < in_d + pad_back; d++) {
      if (d < 0 || d >= in_d) {
        memset(dout_num, pad_value, sizeof(float) * size_out_hw);
        dout_num += size_out_hw;
      } else {
        for (int h = -pad_top; h < in_h + pad_bottom; h++) {
          if (h < 0 || h >= in_h) {
            memset(dout_num, pad_value, sizeof(float) * out_wc);
            dout_num += out_wc;
          } else {
            if (pad_left) {
              memset(dout_num, pad_value, sizeof(float) * pad_leftc);
              dout_num += pad_leftc;
            }
            memcpy(dout_num, din_num, sizeof(float) * in_wc);
            dout_num += in_wc;
            din_num += in_wc;
            if (pad_right) {
              memset(dout_num, pad_value, sizeof(float) * pad_rightc);
              dout_num += pad_rightc;
            }
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ncdhw_reflect(const float* din,
                       float* dout,
                       int n,
                       int c,
                       int in_d,
                       int in_h,
                       int in_w,
                       int out_d,
                       int out_h,
                       int out_w,
                       const int pad_top,
                       const int pad_bottom,
                       const int pad_left,
                       const int pad_right,
                       const int pad_front,
                       const int pad_back) {
  int num = n * c;
  int size_in_hw = in_h * in_w;
  int size_out_hw = out_h * out_w;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, num) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int od = 0; od < out_d; od++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int in_depth = od - pad_front;
          int in_height = oh - pad_top;
          int in_width = ow - pad_left;
          in_depth = std::max(in_depth, -in_depth);
          in_depth = std::min(in_depth, 2 * in_d - in_depth - 2);
          in_height = std::max(in_height, -in_height);
          in_height = std::min(in_height, 2 * in_h - in_height - 2);
          in_width = std::max(in_width, -in_width);
          in_width = std::min(in_width, 2 * in_w - in_width - 2);
          dout_num[od * size_out_hw + oh * out_w + ow] =
              din_num[in_depth * size_in_hw + in_height * in_w + in_width];
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ndhwc_reflect(const float* din,
                       float* dout,
                       int n,
                       int c,
                       int in_d,
                       int in_h,
                       int in_w,
                       int out_d,
                       int out_h,
                       int out_w,
                       const int pad_top,
                       const int pad_bottom,
                       const int pad_left,
                       const int pad_right,
                       const int pad_front,
                       const int pad_back) {
  int in_wc = in_w * c;
  int out_wc = out_w * c;
  int size_in_hw = in_h * in_wc;
  int size_out_hw = out_h * out_wc;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, n) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int od = 0; od < out_d; od++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int in_depth = od - pad_front;
          int in_height = oh - pad_top;
          int in_width = ow - pad_left;
          in_depth = std::max(in_depth, -in_depth);
          in_depth = std::min(in_depth, 2 * in_d - in_depth - 2);
          in_height = std::max(in_height, -in_height);
          in_height = std::min(in_height, 2 * in_h - in_height - 2);
          in_width = std::max(in_width, -in_width);
          in_width = std::min(in_width, 2 * in_w - in_width - 2);
          int out_idx = od * size_out_hw + oh * out_wc + ow * c;
          int in_idx = in_depth * size_in_hw + in_height * in_wc + in_width * c;
          for (int j = 0; j < c; j++) {
            dout_num[out_idx + j] = din_num[in_idx + j];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ncdhw_replicate(const float* din,
                         float* dout,
                         int n,
                         int c,
                         int in_d,
                         int in_h,
                         int in_w,
                         int out_d,
                         int out_h,
                         int out_w,
                         const int pad_top,
                         const int pad_bottom,
                         const int pad_left,
                         const int pad_right,
                         const int pad_front,
                         const int pad_back) {
  int num = n * c;
  int size_in_hw = in_h * in_w;
  int size_out_hw = out_h * out_w;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, num) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int od = 0; od < out_d; od++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int in_depth = std::min(in_d - 1, std::max(od - pad_front, 0));
          int in_height = std::min(in_h - 1, std::max(oh - pad_top, 0));
          int in_width = std::min(in_w - 1, std::max(ow - pad_left, 0));
          dout_num[od * size_out_hw + oh * out_w + ow] =
              din_num[in_depth * size_in_hw + in_height * in_w + in_width];
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ndhwc_replicate(const float* din,
                         float* dout,
                         int n,
                         int c,
                         int in_d,
                         int in_h,
                         int in_w,
                         int out_d,
                         int out_h,
                         int out_w,
                         const int pad_top,
                         const int pad_bottom,
                         const int pad_left,
                         const int pad_right,
                         const int pad_front,
                         const int pad_back) {
  int in_wc = in_w * c;
  int out_wc = out_w * c;
  int size_in_hw = in_h * in_wc;
  int size_out_hw = out_h * out_wc;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, n) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int od = 0; od < out_d; od++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int in_depth = std::min(in_d - 1, std::max(od - pad_front, 0));
          int in_height = std::min(in_h - 1, std::max(oh - pad_top, 0));
          int in_width = std::min(in_w - 1, std::max(ow - pad_left, 0));
          int out_idx = od * size_out_hw + oh * out_wc + ow * c;
          int in_idx = in_depth * size_in_hw + in_height * in_wc + in_width * c;
          for (int j = 0; j < c; j++) {
            dout_num[out_idx + j] = din_num[in_idx + j];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ncdhw_circular(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back) {
  int num = n * c;
  int size_in_hw = in_h * in_w;
  int size_out_hw = out_h * out_w;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, num) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int od = 0; od < out_d; od++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int in_depth = ((od - pad_front) % in_d + in_d) % in_d;
          int in_height = ((oh - pad_top) % in_h + in_h) % in_h;
          int in_width = ((ow - pad_left) % in_w + in_w) % in_w;
          dout_num[od * size_out_hw + oh * out_w + ow] =
              din_num[in_depth * size_in_hw + in_height * in_w + in_width];
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad_ndhwc_circular(const float* din,
                        float* dout,
                        int n,
                        int c,
                        int in_d,
                        int in_h,
                        int in_w,
                        int out_d,
                        int out_h,
                        int out_w,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right,
                        const int pad_front,
                        const int pad_back) {
  int in_wc = in_w * c;
  int out_wc = out_w * c;
  int size_in_hw = in_h * in_wc;
  int size_out_hw = out_h * out_wc;
  int spatial_size_out = size_out_hw * out_d;
  int spatial_size_in = size_in_hw * in_d;
  LITE_PARALLEL_BEGIN(i, tid, n) {
    const float* din_num = din + i * spatial_size_in;
    float* dout_num = dout + i * spatial_size_out;
    for (int od = 0; od < out_d; od++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          int in_depth = ((od - pad_front) % in_d + in_d) % in_d;
          int in_height = ((oh - pad_top) % in_h + in_h) % in_h;
          int in_width = ((ow - pad_left) % in_w + in_w) % in_w;
          int out_idx = od * size_out_hw + oh * out_wc + ow * c;
          int in_idx = in_depth * size_in_hw + in_height * in_wc + in_width * c;
          for (int j = 0; j < c; j++) {
            dout_num[out_idx + j] = din_num[in_idx + j];
          }
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

void pad3d_ncdhw_func(const lite::Tensor* input,
                      lite::Tensor* output,
                      int n,
                      int c,
                      int in_d,
                      int in_h,
                      int in_w,
                      int out_d,
                      int out_h,
                      int out_w,
                      int mode,
                      std::vector<int> pad_h,
                      std::vector<int> pad_w,
                      std::vector<int> pad_d,
                      float pad_value) {
  float* dout = output->mutable_data<float>();
  const float* din = input->data<float>();

  auto output_dims = output->dims();

  if (mode == 0) {
    pad_ncdhw_constant(din,
                       dout,
                       n,
                       c,
                       in_d,
                       in_h,
                       in_w,
                       out_d,
                       out_h,
                       out_w,
                       pad_h[0],
                       pad_h[1],
                       pad_w[0],
                       pad_w[1],
                       pad_d[0],
                       pad_d[1],
                       pad_value);
  } else if (mode == 1) {
    pad_ncdhw_reflect(din,
                      dout,
                      n,
                      c,
                      in_d,
                      in_h,
                      in_w,
                      out_d,
                      out_h,
                      out_w,
                      pad_h[0],
                      pad_h[1],
                      pad_w[0],
                      pad_w[1],
                      pad_d[0],
                      pad_d[1]);
  } else if (mode == 2) {
    pad_ncdhw_replicate(din,
                        dout,
                        n,
                        c,
                        in_d,
                        in_h,
                        in_w,
                        out_d,
                        out_h,
                        out_w,
                        pad_h[0],
                        pad_h[1],
                        pad_w[0],
                        pad_w[1],
                        pad_d[0],
                        pad_d[1]);
  } else if (mode == 3) {
    pad_ncdhw_circular(din,
                       dout,
                       n,
                       c,
                       in_d,
                       in_h,
                       in_w,
                       out_d,
                       out_h,
                       out_w,
                       pad_h[0],
                       pad_h[1],
                       pad_w[0],
                       pad_w[1],
                       pad_d[0],
                       pad_d[1]);
  } else {
    LOG(ERROR) << "ERROR: unknown pad mode " << mode;
  }
}

void pad3d_ndhwc_func(const lite::Tensor* input,
                      lite::Tensor* output,
                      int n,
                      int c,
                      int in_d,
                      int in_h,
                      int in_w,
                      int out_d,
                      int out_h,
                      int out_w,
                      int mode,
                      std::vector<int> pad_h,
                      std::vector<int> pad_w,
                      std::vector<int> pad_d,
                      float pad_value) {
  float* dout = output->mutable_data<float>();
  const float* din = input->data<float>();

  auto output_dims = output->dims();

  if (mode == 0) {
    pad_ndhwc_constant(din,
                       dout,
                       n,
                       c,
                       in_d,
                       in_h,
                       in_w,
                       out_d,
                       out_h,
                       out_w,
                       pad_h[0],
                       pad_h[1],
                       pad_w[0],
                       pad_w[1],
                       pad_d[0],
                       pad_d[1],
                       pad_value);
  } else if (mode == 1) {
    pad_ndhwc_reflect(din,
                      dout,
                      n,
                      c,
                      in_d,
                      in_h,
                      in_w,
                      out_d,
                      out_h,
                      out_w,
                      pad_h[0],
                      pad_h[1],
                      pad_w[0],
                      pad_w[1],
                      pad_d[0],
                      pad_d[1]);
  } else if (mode == 2) {
    pad_ndhwc_replicate(din,
                        dout,
                        n,
                        c,
                        in_d,
                        in_h,
                        in_w,
                        out_d,
                        out_h,
                        out_w,
                        pad_h[0],
                        pad_h[1],
                        pad_w[0],
                        pad_w[1],
                        pad_d[0],
                        pad_d[1]);
  } else if (mode == 3) {
    pad_ndhwc_circular(din,
                       dout,
                       n,
                       c,
                       in_d,
                       in_h,
                       in_w,
                       out_d,
                       out_h,
                       out_w,
                       pad_h[0],
                       pad_h[1],
                       pad_w[0],
                       pad_w[1],
                       pad_d[0],
                       pad_d[1]);
  } else {
    LOG(ERROR) << "ERROR: unknown pad mode " << mode;
  }
}
}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
