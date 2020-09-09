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

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/lrn_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

/**
 * @brief get sum of x^2 between channels [size elements]
 *
 * @tparam dtype
 * @param input
 * @param channel_id: the c-th channel within n-th graph.
 * @param offset_within_channel: the pixel's offset within a channel.
 * @param offset_num: the first address of n-th graph.
 * @param c
 * @param h
 * @param w
 * @param size
 * @return dtype
 */
template <typename dtype>
dtype lrn_square(const dtype* input,
                 int channel_id,
                 int offset_within_channel,
                 int offset_num,
                 int c,
                 int h,
                 int w,
                 int size) {
  int pre_pad = (size - 1) / 2;
  dtype res = 0;
  const dtype* src = input + offset_num;

  // handle left channels with padding situation.
  if (channel_id - pre_pad < 0) {
    for (int i = 0; i <= channel_id; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  // handle left channels.
  if (channel_id - pre_pad >= 0) {
    for (int i = channel_id - pre_pad; i <= channel_id; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  // handle right channels.
  if (channel_id + pre_pad < c) {
    for (int i = channel_id + 1; i <= channel_id + pre_pad; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  // handle right channels with padding situation.
  if (channel_id + pre_pad >= c && channel_id + 1 < c) {
    for (int i = channel_id + 1; i < c; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  return res;
}

template <typename dtype>
void lrn_compute_ref(const operators::LrnParam& param) {
  const dtype* x_data = param.X->data<const dtype>();
  dtype* out_data = param.Out->mutable_data<dtype>();
  auto x_dims = param.X->dims();
  int local_size = param.n;
  float alpha = param.alpha;
  float beta = param.beta;
  float k = param.k;
  std::string norm_region = param.norm_region;

  int N = x_dims[0];
  int C = x_dims[1];
  int H = x_dims[2];
  int W = x_dims[3];

  int pre_pad = (local_size - 1) / 2;
  int offset_num = 0;
  int offset_within_channel = 0;
  int dst_id;

  dtype square;

  for (int n = 0; n < N; ++n) {
    offset_num = n * C * H * W;

    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          offset_within_channel = h * W + w;
          dst_id = offset_num + c * H * W + offset_within_channel;
          square = lrn_square<dtype>(x_data,
                                     c,
                                     offset_within_channel,
                                     offset_num,
                                     C,
                                     H,
                                     W,
                                     local_size);
          out_data[dst_id] = x_data[dst_id] * pow(k + alpha * square, -beta);
        }
      }
    }
  }
}

TEST(lrn_arm, retrive_op) {
  auto lrn = KernelRegistry::Global().Create("lrn");
  ASSERT_FALSE(lrn.empty());
  ASSERT_TRUE(lrn.front());
}

TEST(lrn_arm, init) {
  LrnCompute lrn;
  ASSERT_EQ(lrn.precision(), PRECISION(kFloat));
  ASSERT_EQ(lrn.target(), TARGET(kARM));
}

TEST(lrn_arm, compute) {
  LrnCompute lrn;
  operators::LrnParam param;
  lite::Tensor x, output, output_ref;

  int local_size = 5;
  float alpha = 1.0f;
  float beta = 0.75;
  float k = 1.0f;
  std::string norm_region = "AcrossChannels";
  for (int w : {1, 2, 4, 8}) {
    for (int h : {1, 2, 4, 8}) {
      for (int c : {1, 2, 3, 4}) {
        for (int n : {1, 2, 3, 4}) {
          auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
          x.Resize(x_dim);
          output.Resize(x_dim);
          output_ref.Resize(x_dim);
          auto* x_data = x.mutable_data<float>();
          auto* output_data = output.mutable_data<float>();
          auto* output_ref_data = output_ref.mutable_data<float>();
          for (int i = 0; i < x_dim.production(); i++) {
            x_data[i] = i;
          }
          param.X = &x;
          param.Out = &output;
          param.n = local_size;
          param.alpha = alpha;
          param.beta = beta;
          param.k = k;
          param.norm_region = norm_region;
          lrn.SetParam(param);
          lrn.Run();
          param.Out = &output_ref;
          lrn_compute_ref<float>(param);
          for (int i = 0; i < output.dims().production(); i++) {
            EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
          }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(lrn, kARM, kFloat, kNCHW, def);
