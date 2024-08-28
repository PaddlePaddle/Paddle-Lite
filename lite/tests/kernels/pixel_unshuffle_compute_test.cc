// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class PixelUnshuffleComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  int downscale_factor_ = 2;
  DDim dims_{{2, 4, 4, 4}};

 public:
  PixelUnshuffleComputeTester(const Place& place,
                              const std::string& alias,
                              int downscale_factor,
                              int n,
                              int c,
                              int h,
                              int w)
      : TestCase(place, alias), downscale_factor_(downscale_factor) {
    dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    const int64_t batch_size = dims_[0];
    const int64_t in_channels = dims_[1];
    const int64_t in_height = dims_[2];
    const int64_t in_width = dims_[3];

    const int64_t out_channels =
        in_channels * (downscale_factor_ * downscale_factor_);
    const int64_t out_height = in_height / downscale_factor_;
    const int64_t out_width = in_width / downscale_factor_;

    int64_t nchw[] = {batch_size, out_channels, out_height, out_width};
    std::vector<int64_t> output_shape(nchw, nchw + 4);
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < in_channels; ++c) {
        for (int h = 0; h < in_height; ++h) {
          for (int w = 0; w < in_width; ++w) {
            int out_c = c * (downscale_factor_ * downscale_factor_) +
                        (h % downscale_factor_) * downscale_factor_ +
                        (w % downscale_factor_);
            int out_h = h / downscale_factor_;
            int out_w = w / downscale_factor_;
            output_data[n * out_channels * out_height * out_width +
                        out_c * out_height * out_width + out_h * out_width +
                        out_w] =
                x_data[n * in_channels * in_height * in_width +
                       c * in_height * in_width + h * in_width + w];
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("pixel_unshuffle");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("downscale_factor", downscale_factor_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

TEST(PixelUnshuffle, precision) {
  Place place = TARGET(kHost);
  LOG(INFO) << "test pixel_unshuffle op";

  for (int downscale_factor : {2, 3}) {
    for (int n : {1, 3}) {
      for (int c : {36, 72, 144}) {
        for (int h : {6, 18}) {
          for (int w : {6, 18}) {
            LOG(INFO) << "n: " << n << " c: " << c << " h: " << h << " w: " << w
                      << " downscale_factor: " << downscale_factor;
            std::unique_ptr<arena::TestCase> tester(
                new PixelUnshuffleComputeTester(
                    place, "def", downscale_factor, n, c, h, w));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}
}  // namespace lite
}  // namespace paddle
