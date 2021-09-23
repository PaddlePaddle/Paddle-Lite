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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {

class AffineGridComputeTester : public arena::TestCase {
 protected:
  std::string theta_ = "x";
  std::string out_shape_ = "out_shape";
  std::string output_ = "output";
  std::vector<int> output_shape_{0};
  std::vector<int> out_shape_data_{0};
  bool align_corners_ = true;
  DDim x_dims_{{1, 2, 3}};
  DDim out_shape_dims_{{4}};

 public:
  AffineGridComputeTester(const Place& place,
                          const std::string& alias,
                          std::vector<int> out_shape_data,
                          bool align_corners,
                          std::vector<int> output_shape)
      : TestCase(place, alias),
        output_shape_(output_shape),
        out_shape_data_(out_shape_data) {
    align_corners_ = align_corners;
    x_dims_[0] = out_shape_data[0];
  }

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(theta_);
    auto din = x->data<float>();

    int N = x_dims_[0];
    int H, W;
    if (output_shape_.size() == 0) {
      H = out_shape_data_[2];
      W = out_shape_data_[3];
    } else {
      H = output_shape_[2];
      W = output_shape_[3];
    }
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(std::vector<int64_t>({N, H, W, 2}));
    auto dout = out->mutable_data<float>();

    std::vector<float> vvh(H);
    float* vh = vvh.data();
    std::vector<float> vvw(W);
    float* vw = vvw.data();

    int out_size = H * W * 3;
    float scale = 2 / (static_cast<float>(H) - 1);
    float start = -1.0f;
    if (!align_corners_) {
      scale = 2 / static_cast<float>(H);
      start *= (static_cast<float>(H) - 1) / static_cast<float>(H);
    }
    for (int i = 0; i < H; i++) {
      vh[i] = start + scale * i;
    }

    scale = 2 / (static_cast<float>(W) - 1);
    start = -1.0f;
    if (!align_corners_) {
      scale = 2 / static_cast<float>(W);
      start *= (static_cast<float>(W) - 1) / static_cast<float>(W);
    }
    for (int i = 0; i < W; i++) {
      vw[i] = start + i * scale;
    }
    std::vector<float> vhw3(out_size);
    float* hw3 = vhw3.data();

    for (int i = 0; i < out_size; i += 3) {
      hw3[i] = 1;
      hw3[i + 1] = 1;
      hw3[i + 2] = 1;
    }

    for (int i = 0; i < H * W; i++) {
      hw3[i * 3 + 1] = vh[i / W];
    }
    for (int i = 0; i < H * W; i++) {
      hw3[i * 3] = vw[i % W];
    }
    const float* bias = nullptr;
    for (int i = 0; i < out->dims().production(); i++) {
      dout[i] = 0;
    }
    for (int i = 0; i < N; i++) {
      basic_gemm(false,
                 true,
                 H * W,
                 2,
                 3,
                 1.f,
                 hw3,
                 3,
                 din,
                 3,
                 0.f,
                 dout,
                 2,
                 bias,
                 false,
                 false);
      din += 6;
      dout += H * W * 2;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("affine_grid");
    op_desc->SetInput("Theta", {theta_});
    op_desc->SetAttr("output_shape", output_shape_);
    op_desc->SetInput("OutputShape", {out_shape_});
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("align_corners", align_corners_);
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(theta_, x_dims_, x_data.data());
    SetCommonTensor(out_shape_, out_shape_dims_, out_shape_data_.data());
  }
};

TEST(AffineGrid, precision) {
  LOG(INFO) << "test affine_grid op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (int n : {1, 5}) {
    for (int c : {1, 3}) {
      for (int h : {3, 10}) {
        for (int w : {3, 10}) {
          for (bool align_corners : {false, true}) {
            for (bool output_shape : {false, true}) {
              std::vector<int> out_shape;
              std::vector<int> out_shape_data = {n, c, h, w};
              if (output_shape) {
                out_shape = {n, c, h * 2, w * 2};
                std::unique_ptr<arena::TestCase> tester(
                    new AffineGridComputeTester(place,
                                                "def",
                                                out_shape_data,
                                                align_corners,
                                                out_shape));
                arena::Arena arena(std::move(tester), place, 2e-5);
                arena.TestPrecision();
              } else {
                std::unique_ptr<arena::TestCase> tester(
                    new AffineGridComputeTester(place,
                                                "def",
                                                out_shape_data,
                                                align_corners,
                                                out_shape));
                arena::Arena arena(std::move(tester), place, 2e-5);
                arena.TestPrecision();
              }
            }
          }
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
