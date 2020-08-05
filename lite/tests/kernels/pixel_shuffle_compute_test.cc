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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class PixelShuffleComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  int upscale_factor_ = 3;
  DDim dims_{{2, 27, 20, 30}};

 public:
  PixelShuffleComputeTester(const Place& place,
                            const std::string& alias,
                            int upscale_factor,
                            int n,
                            int c,
                            int h,
                            int w)
      : TestCase(place, alias), upscale_factor_(upscale_factor) {
    dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    const int64_t batch_size = dims_[0];
    const int64_t out_channels = dims_[1] / (upscale_factor_ * upscale_factor_);
    const int64_t out_height = dims_[2] * upscale_factor_;
    const int64_t out_width = dims_[3] * upscale_factor_;

    int64_t nchw[] = {batch_size, out_channels, out_height, out_width};
    std::vector<int64_t> output_shape(nchw, nchw + 4);
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    for (int nc = 0; nc < batch_size * out_channels; nc++) {
      const float* inptr = x_data + nc * out_height * out_width;
      float* outptr_nc = output_data + nc * out_height * out_width;

      for (int sh = 0; sh < upscale_factor_; sh++) {
        for (int sw = 0; sw < upscale_factor_; sw++) {
          float* outptr = outptr_nc + sh * out_width + sw;
          for (int h = 0; h < dims_[2]; h++) {
            for (int w = 0; w < dims_[3]; w++) {
              outptr[0] = inptr[0];
              inptr++;
              outptr += upscale_factor_;
            }
            outptr += (upscale_factor_ - 1) * out_width;
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("pixel_shuffle");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("upscale_factor", upscale_factor_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

TEST(PixelShuffle, precision) {
  LOG(INFO) << "test pixel_shuffle op";
#ifdef LITE_WITH_ARM
  LOG(INFO) << "test pixel_shuffle arm";
  Place place(TARGET(kARM));

  for (int upscale_factor : {1, 2, 3, 4, 5}) {
    for (int n : {1, 3}) {
      for (int c : {3 * upscale_factor * upscale_factor,
                    6 * upscale_factor * upscale_factor}) {
        for (int h : {9, 18}) {
          for (int w : {9, 18}) {
            std::unique_ptr<arena::TestCase> tester(
                new PixelShuffleComputeTester(
                    place, "def", upscale_factor, n, c, h, w));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
