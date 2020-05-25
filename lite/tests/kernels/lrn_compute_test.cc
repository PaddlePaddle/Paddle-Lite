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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

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

class LrnComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  float alpha_ = 1.;
  float beta_ = 0.75;
  float k_ = 1.;
  int local_size_ = 5;
  DDim dims_{{2, 4, 3, 8}};
  std::string norm_region_{"AcrossChannels"};

 public:
  LrnComputeTester(const Place& place,
                   const std::string& alias,
                   float alpha,
                   float beta,
                   float k,
                   int local_size,
                   std::string norm_region)
      : TestCase(place, alias),
        alpha_(alpha),
        beta_(beta),
        k_(k),
        local_size_(local_size),
        norm_region_(norm_region) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    int N = dims_[0];
    int C = dims_[1];
    int H = dims_[2];
    int W = dims_[3];

    int offset_num = 0;
    int offset_within_channel = 0;
    int dst_id;

    float square;

    for (int n = 0; n < N; ++n) {
      offset_num = n * C * H * W;
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            offset_within_channel = h * W + w;
            dst_id = offset_num + c * H * W + offset_within_channel;
            square = lrn_square<float>(x_data,
                                       c,
                                       offset_within_channel,
                                       offset_num,
                                       C,
                                       H,
                                       W,
                                       local_size_);
            out_data[dst_id] =
                x_data[dst_id] * pow(k_ + alpha_ * square, -beta_);
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("lrn");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("alpha", alpha_);
    op_desc->SetAttr("beta", beta_);
    op_desc->SetAttr("n", local_size_);
    op_desc->SetAttr("k", k_);
    op_desc->SetAttr("norm_region", norm_region_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

void test_lrn(Place place) {
  for (float alpha : {0.9, 1., 1.1}) {
    for (float beta : {0.5, 0.75, 1.}) {
      for (float k : {0.9, 1., 1.1}) {
        for (int local_size : {4, 5, 7}) {
          for (std::string norm_region : {"AcrossChannels"}) {
            std::unique_ptr<arena::TestCase> tester(new LrnComputeTester(
                place, "def", alpha, beta, k, local_size, norm_region));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(Lrn, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_lrn(place);
#endif
}

}  // namespace lite
}  // namespace paddle
