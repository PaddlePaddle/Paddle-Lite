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

// We refer https://github.com/andravin/wincnn to access the winograd transform
// matrixs

#ifdef CONV_OP
#ifdef __aarch64__

#include "operators/math/winograd/winograd_transform.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <>
void winograd_transform_weight<8, 3>(const framework::Tensor &weight,
                                     framework::Tensor *output) {
  // weight shape is [out_channel, in_channel, kernel_h, kernel_w]
  int out_channel = weight.dims()[0];
  int in_channel = weight.dims()[1];
  // reshape and alloc transformed weight
  framework::DDim transformed_shape =
      framework::make_ddim(std::vector<int>{out_channel, in_channel, 64});
  float *outptr = output->mutable_data<float>(transformed_shape);
  const float *inptr = weight.data<float>();
  for (int oc = 0; oc < out_channel; ++oc) {
    for (int ic = 0; ic < in_channel; ++ic) {
      size_t offset = oc * in_channel + ic;
      float *kout = outptr + offset * 64;
      const float *k = inptr + offset * 9;

      float gw[3][8];
      for (int i = 0; i < 3; ++i, k += 3) {
        float g0 = k[0];
        float g1 = k[1];
        float g2 = k[2];
        float d0 = g0 + g2;
        float d1 = g0 + 4 * g2;
        float d2 = g2 + 4 * g0;
        float d3 = 2 * g1;
        gw[i][0] = g0;
        gw[i][1] = -2.f / 9 * (d0 + g1);   // -2.f/9 * (g0 + g1 + g2)
        gw[i][2] = -2.f / 9 * (d0 - g1);   // -2.f/9 * (g0 - g1 + g2)
        gw[i][3] = 1.f / 90 * (d1 + d3);   // 1.f/90 * (g0 + 2 * g1 + 4 * g2)
        gw[i][4] = 1.f / 90 * (d1 - d3);   // 1.f/90 * (g0 - 2 * g1 + 4 * g2)
        gw[i][5] = 1.f / 180 * (d2 + d3);  // 1.f/180 * (4 * g0 + 2 * g1 + g2)
        gw[i][6] = 1.f / 180 * (d2 - d3);  // 1.f/180 * (4 * g0 - 2 * g1 + g2)
        gw[i][7] = g2;
      }
      for (int i = 0; i < 8; ++i, kout += 8) {
        float g0 = gw[0][i];
        float g1 = gw[1][i];
        float g2 = gw[2][i];
        float d0 = g0 + g2;
        float d1 = g0 + 4 * g2;
        float d2 = g2 + 4 * g0;
        float d3 = 2 * g1;
        kout[0] = g0;
        kout[1] = -2.f / 9 * (d0 + g1);   // -2.f/9 * (k0 + k1 + k2)
        kout[2] = -2.f / 9 * (d0 - g1);   // -2.f/9 * (k0 - k1 + k2)
        kout[3] = 1.f / 90 * (d1 + d3);   // 1.f/90 * (k0 + 2 * k1 + 4 * k2)
        kout[4] = 1.f / 90 * (d1 - d3);   // 1.f/90 * (k0 - 2 * k1 + 4 * k2)
        kout[5] = 1.f / 180 * (d2 + d3);  // 8.f/45 * (4 * k0 + 2 * k1 + k2)
        kout[6] = 1.f / 180 * (d2 - d3);  // 8.f/45 * (4 * k0 - 2 * k1 + k2)
        kout[7] = g2;
      }
    }
  }
}

template <>
void winograd_transform_input<8, 3>(const framework::Tensor &input,
                                    framework::Tensor *output) {
  // tile input to [c, roundup(h/6), roundup(w/6), 64] and do transformation
  int channel = input.dims()[1];
  int height = input.dims()[2];
  int width = input.dims()[3];
  int h_tiles = (height + 3) / 6;  // (height + 5 - 2) / 6
  int w_tiles = (width + 3) / 6;   // (width + 5 - 2) / 6
  framework::DDim transformed_shape =
      framework::make_ddim(std::vector<int>{channel, h_tiles, w_tiles, 64});
  float *outptr = output->mutable_data<float>(transformed_shape);
  memset(outptr, 0, channel * h_tiles * w_tiles * 64 * sizeof(float));
  const float *inptr = input.data<float>();
  // pack input to tiles
  for (int c = 0; c < channel; ++c) {
    int inter_h = (height - 2) / 6;
    int inter_w = (width - 2) / 6;
    int remain_h = height - (inter_h * 6);
    int remain_w = width - (inter_w * 6);
    const float *in0 = inptr + c * height * width;
    const float *in1 = in0 + width;
    const float *in2 = in1 + width;
    const float *in3 = in2 + width;
    const float *in4 = in3 + width;
    const float *in5 = in4 + width;
    const float *in6 = in5 + width;
    const float *in7 = in6 + width;
    float *out = outptr + c * h_tiles * w_tiles * 64;

    for (int h = 0; h < inter_h; ++h) {
      for (int w = 0; w < inter_w; ++w) {
        memcpy(out, in0, 8 * sizeof(float));
        memcpy(out + 8, in1, 8 * sizeof(float));
        memcpy(out + 16, in2, 8 * sizeof(float));
        memcpy(out + 24, in3, 8 * sizeof(float));
        memcpy(out + 32, in4, 8 * sizeof(float));
        memcpy(out + 40, in5, 8 * sizeof(float));
        memcpy(out + 48, in6, 8 * sizeof(float));
        memcpy(out + 56, in7, 8 * sizeof(float));
        in0 += 6;
        in1 += 6;
        in2 += 6;
        in3 += 6;
        in4 += 6;
        in5 += 6;
        in6 += 6;
        in7 += 6;
        out += 64;
      }
      // remain width
      if (remain_w > 2) {
        memcpy(out, in0, remain_w * sizeof(float));
        memcpy(out + 8, in1, remain_w * sizeof(float));
        memcpy(out + 16, in2, remain_w * sizeof(float));
        memcpy(out + 24, in3, remain_w * sizeof(float));
        memcpy(out + 32, in4, remain_w * sizeof(float));
        memcpy(out + 40, in5, remain_w * sizeof(float));
        memcpy(out + 48, in6, remain_w * sizeof(float));
        memcpy(out + 56, in7, remain_w * sizeof(float));
        out += 64;
      }
      in0 += 5 * width + remain_w;
      in1 += 5 * width + remain_w;
      in2 += 5 * width + remain_w;
      in3 += 5 * width + remain_w;
      in4 += 5 * width + remain_w;
      in5 += 5 * width + remain_w;
      in6 += 5 * width + remain_w;
      in7 += 5 * width + remain_w;
    }
    // remain height
    if (remain_h > 2) {
      for (int w = 0; w < inter_w; ++w) {
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out + rh * 8, in0 + rh * width, 8 * sizeof(float));
        }
        out += 64;
        in0 += 6;
      }
      // remain width
      if (remain_w > 2) {
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out + rh * 8, in0 + rh * width, remain_w * sizeof(float));
        }
      }
    }
  }
  // transform tiles, compute B_T * d(c, b) * B
  for (int c = 0; c < channel; ++c) {
    for (int tile = 0; tile < h_tiles * w_tiles; ++tile) {
      float *out = outptr + (c * h_tiles * w_tiles + tile) * 64;
      // compute B_T * d(c, b)
      float bd[8][8];
      for (int i = 0; i < 8; ++i) {
        float d0 = out[8 * i + 0];
        float d1 = out[8 * i + 1];
        float d2 = out[8 * i + 2];
        float d3 = out[8 * i + 3];
        float d4 = out[8 * i + 4];
        float d5 = out[8 * i + 5];
        float d6 = out[8 * i + 6];
        float d7 = out[8 * i + 7];

        bd[i][0] = d0 - d6 + (d4 - d2) * 5.25;
        float v1 = d2 - 4.25 * d4 + d6;
        float v2 = d1 - 4.25 * d3 + d5;
        // d1 + d2 - 4.25 * d3 - 4.25 * d4 + d5 + d6
        bd[i][1] = v1 + v2;
        // -d1 + d2 + 4.25 * d3 - 4.25 * d4 - d5 + d6
        bd[i][2] = v1 - v2;
        v1 = 0.25 * d2 - 1.25 * d4 + d6;
        v2 = 0.5 * d1 - 2.5 * d3 + 2 * d5;
        // 0.5 * d1 + 0.25 * d2 - 2.5 * d3 - 1.25 * d4 + 2 * d5 + d6
        bd[i][3] = v1 + v2;
        // -0.5 * d1 + 0.25 * d2 + 2.5 * d3 - 1.25 * d4 - 2 * d5 + d6
        bd[i][4] = v1 - v2;
        v1 = 4 * d2 - 5 * d4 + d6;
        v2 = 2 * d1 - 2.5 * d3 + 0.5 * d5;
        // 2 * d1 + 4 * d2 - 2.5 * d3 - 5 * d4 + 0.5 * d5 + d6
        bd[i][5] = v1 + v2;
        // -2 * d1 + 4 * d2 + 2.5 * d3 - 5 * d4 - 0.5 * d5 + d6
        bd[i][6] = v1 - v2;
        bd[i][7] = d7 - d1 + (d3 - d5) * 5.25;
      }
      // compute B_T * d(c, b) * B
      for (int i = 0; i < 8; ++i, out += 8) {
        float d0 = bd[0][i];
        float d1 = bd[1][i];
        float d2 = bd[2][i];
        float d3 = bd[3][i];
        float d4 = bd[4][i];
        float d5 = bd[5][i];
        float d6 = bd[6][i];
        float d7 = bd[7][i];

        out[0] = d0 - d6 + (d4 - d2) * 5.25;
        float v1 = d2 - 4.25 * d4 + d6;
        float v2 = d1 - 4.25 * d3 + d5;
        // d1 + d2 - 4.25 * d3 - 4.25 * d4 + d5 + d6
        out[1] = v1 + v2;
        // -d1 + d2 + 4.25 * d3 - 4.25 * d4 - d5 + d6
        out[2] = v1 - v2;
        v1 = 0.25 * d2 - 1.25 * d4 + d6;
        v2 = 0.5 * d1 - 2.5 * d3 + 2 * d5;
        // 0.5 * d1 + 0.25 * d2 - 2.5 * d3 - 1.25 * d4 + 2 * d5 + d6
        out[3] = v1 + v2;
        // -0.5 * d1 + 0.25 * d2 + 2.5 * d3 - 1.25 * d4 - 2 * d5 + d6
        out[4] = v1 - v2;
        v1 = 4 * d2 - 5 * d4 + d6;
        v2 = 2 * d1 - 2.5 * d3 + 0.5 * d5;
        // 2 * d1 + 4 * d2 - 2.5 * d3 - 5 * d4 + 0.5 * d5 + d6
        out[5] = v1 + v2;
        // -2 * d1 + 4 * d2 + 2.5 * d3 - 5 * d4 - 0.5 * d5 + d6
        out[6] = v1 - v2;
        out[7] = d7 - d1 + (d3 - d5) * 5.25;
      }
    }
  }
}

template <>
void winograd_transform_output<8, 3>(const framework::Tensor &input,
                                     const framework::Tensor &weight,
                                     framework::Tensor *output) {
  // input shape is [in_channel, h_tiles, w_tiles, 64]
  // weight shape is [out_channel, in_channel, 64]
  int in_channel = input.dims()[0];
  int h_tiles = input.dims()[1];
  int w_tiles = input.dims()[2];
  int tiles = h_tiles * w_tiles;
  int out_channel = weight.dims()[0];
  // compute U*V first
  framework::Tensor output_m;
  framework::DDim shape =
      framework::make_ddim(std::vector<int>{out_channel, tiles, 64});
  float *output_m_ptr = output_m.mutable_data<float>(shape);
  memset(output_m_ptr, 0, output_m.numel() * sizeof(float));
  const float *input_ptr = input.data<float>();
  const float *weight_ptr = weight.data<float>();
  for (int i = 0; i < out_channel; ++i) {
    for (int j = 0; j < tiles; ++j) {
      const float *w_ptr = weight_ptr + i * in_channel * 64;
      const float *in_ptr = input_ptr + j * 64;
      float *m_ptr = output_m_ptr + (i * tiles + j) * 64;
      for (int c = 0; c < in_channel; ++c) {
        for (int k = 0; k < 64; ++k) {
          m_ptr[k] += w_ptr[k] * in_ptr[k];
        }
        w_ptr += 64;
        in_ptr += tiles * 64;
      }
    }
  }

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int tile = 0; tile < tiles; ++tile) {
      float *m = output_m_ptr + (oc * tiles + tile) * 64;
      // compute A_T * m
      float am[6][8];
      for (int i = 0; i < 8; ++i) {
        float d0 = m[i * 8 + 0];
        float d1 = m[i * 8 + 1];
        float d2 = m[i * 8 + 2];
        float d3 = m[i * 8 + 3];
        float d4 = m[i * 8 + 4];
        float d5 = m[i * 8 + 5];
        float d6 = m[i * 8 + 6];
        float d7 = m[i * 8 + 7];
        float v0 = d1 + d2;
        float v1 = d1 - d2;
        float v2 = d3 + d4;
        float v3 = d3 - d4;
        float v4 = d5 + d6;
        float v5 = d5 - d6;

        am[0][i] = d0 + v0 + v2 + 32 * v4;
        am[1][i] = v1 + 2 * v3 + 16 * v5;
        am[2][i] = v0 + 4 * v2 + 8 * v4;
        am[3][i] = v1 + 8 * v3 + 4 * v5;
        am[4][i] = v0 + 16 * v2 + 2 * v4;
        am[5][i] = v1 + 32 * v3 + v5 + d7;
      }
      // compute A_T * m * A
      for (int i = 0; i < 6; ++i, m += 8) {
        float d0 = am[i][0];
        float d1 = am[i][1];
        float d2 = am[i][2];
        float d3 = am[i][3];
        float d4 = am[i][4];
        float d5 = am[i][5];
        float d6 = am[i][6];
        float d7 = am[i][7];
        float v0 = d1 + d2;
        float v1 = d1 - d2;
        float v2 = d3 + d4;
        float v3 = d3 - d4;
        float v4 = d5 + d6;
        float v5 = d5 - d6;

        m[0] = d0 + v0 + v2 + 32 * v4;
        m[1] = v1 + 2 * v3 + 16 * v5;
        m[2] = v0 + 4 * v2 + 8 * v4;
        m[3] = v1 + 8 * v3 + 4 * v5;
        m[4] = v0 + 16 * v2 + 2 * v4;
        m[5] = v1 + 32 * v3 + v5 + d7;
      }
    }
  }

  int out_h = output->dims()[2];
  int out_w = output->dims()[3];
  float *output_ptr = output->mutable_data<float>();
  // copy valid region to final output
  for (int oc = 0; oc < out_channel; ++oc) {
    int inter_h = out_h / 6;
    int inter_w = out_w / 6;
    int remain_h = out_h - inter_h * 6;
    int remain_w = out_w - inter_w * 6;

    float *out_ptr0 = output_ptr + oc * out_h * out_w;
    float *out_ptr1 = out_ptr0 + out_w;
    float *out_ptr2 = out_ptr1 + out_w;
    float *out_ptr3 = out_ptr2 + out_w;
    float *out_ptr4 = out_ptr3 + out_w;
    float *out_ptr5 = out_ptr4 + out_w;
    const float *m_ptr = output_m_ptr + oc * tiles * 64;
    for (int tile_h = 0; tile_h < inter_h; ++tile_h) {
      for (int tile_w = 0; tile_w < inter_w; ++tile_w) {
        const float *m = m_ptr + (tile_h * w_tiles + tile_w) * 64;
        memcpy(out_ptr0, m, 6 * sizeof(float));
        memcpy(out_ptr1, m + 8, 6 * sizeof(float));
        memcpy(out_ptr2, m + 16, 6 * sizeof(float));
        memcpy(out_ptr3, m + 24, 6 * sizeof(float));
        memcpy(out_ptr4, m + 32, 6 * sizeof(float));
        memcpy(out_ptr5, m + 40, 6 * sizeof(float));
        out_ptr0 += 6;
        out_ptr1 += 6;
        out_ptr2 += 6;
        out_ptr3 += 6;
        out_ptr4 += 6;
        out_ptr5 += 6;
      }
      // remain w
      if (remain_w > 0) {
        const float *m = m_ptr + (tile_h * w_tiles + inter_w) * 64;
        memcpy(out_ptr0, m, remain_w * sizeof(float));
        memcpy(out_ptr1, m + 8, remain_w * sizeof(float));
        memcpy(out_ptr2, m + 16, remain_w * sizeof(float));
        memcpy(out_ptr3, m + 24, remain_w * sizeof(float));
        memcpy(out_ptr4, m + 32, remain_w * sizeof(float));
        memcpy(out_ptr5, m + 40, remain_w * sizeof(float));
        out_ptr0 += remain_w;
        out_ptr1 += remain_w;
        out_ptr2 += remain_w;
        out_ptr3 += remain_w;
        out_ptr4 += remain_w;
        out_ptr5 += remain_w;
      }
      out_ptr0 += 5 * out_w;
      out_ptr1 += 5 * out_w;
      out_ptr2 += 5 * out_w;
      out_ptr3 += 5 * out_w;
      out_ptr4 += 5 * out_w;
      out_ptr5 += 5 * out_w;
    }
    // remain h
    if (remain_h > 0) {
      for (int tile_w = 0; tile_w < inter_w; ++tile_w) {
        const float *m = m_ptr + (inter_h * w_tiles + tile_w) * 64;
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out_ptr0 + rh * out_w, m + rh * 8, 6 * sizeof(float));
        }
        out_ptr0 += 6;
      }
      if (remain_w > 0) {
        const float *m = m_ptr + (inter_h * w_tiles + inter_w) * 64;
        for (int rh = 0; rh < remain_h; ++rh) {
          memcpy(out_ptr0 + rh * out_w, m + rh * 8, remain_w * sizeof(float));
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __aarch64__
#endif  // CONV_OP
