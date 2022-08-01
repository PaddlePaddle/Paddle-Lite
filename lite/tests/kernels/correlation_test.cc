// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/backends/host/math/pad2d.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class CorrelationComputeTester : public arena::TestCase {
 protected:
  std::string input1_ = "x1";
  std::string input2_ = "x2";
  std::string output_ = "out";
  DDim x_dims_{{2, 3, 4, 5}};
  int pad_size_ = 4;
  int kernel_size_ = 1;
  int max_displacement_ = 4;
  int stride1_ = 1;
  int stride2_ = 1;
  int corr_type_multiply_ = 1;

 public:
  CorrelationComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto x1 = scope->FindTensor(input1_);
    auto x2 = scope->FindTensor(input2_);
    Tensor x1_pad;
    Tensor x2_pad;
    std::vector<int64_t> x_pad_shape = x_dims_.Vectorize();
    x_pad_shape[2] += 2 * pad_size_;
    x_pad_shape[3] += 2 * pad_size_;
    x1_pad.Resize(x_pad_shape);
    x2_pad.Resize(x_pad_shape);
    auto* x1_pad_data = x1_pad.mutable_data<float>();
    auto* x2_pad_data = x2_pad.mutable_data<float>();

    lite::host::math::Pad2DConstNCHW(x1->data<float>(),
                                     x_dims_[0],
                                     x_dims_[1],
                                     x_dims_[2],
                                     x_dims_[3],
                                     x_pad_shape[2],
                                     x_pad_shape[3],
                                     pad_size_,
                                     pad_size_,
                                     0.f,
                                     x1_pad_data);
    lite::host::math::Pad2DConstNCHW(x2->data<float>(),
                                     x_dims_[0],
                                     x_dims_[1],
                                     x_dims_[2],
                                     x_dims_[3],
                                     x_pad_shape[2],
                                     x_pad_shape[3],
                                     pad_size_,
                                     pad_size_,
                                     0.f,
                                     x2_pad_data);

    auto* out = scope->NewTensor(output_);
    const int B = x_dims_[0];
    const int H = x_dims_[2];
    const int W = x_dims_[3];
    const int d = max_displacement_ / stride2_;
    const int D = 2 * d + 1;
    const std::vector<int64_t> out_shape{static_cast<int64_t>(B),
                                         static_cast<int64_t>(D * D),
                                         static_cast<int64_t>(H),
                                         static_cast<int64_t>(W)};
    out->Resize(out_shape);
    auto* out_data = out->mutable_data<float>();

    const int ic = x_pad_shape[1];
    const int ih = x_pad_shape[2];
    const int iw = x_pad_shape[3];
    const int K = kernel_size_;

    const int oc = out_shape[1];
    const int oh = out_shape[2];
    const int ow = out_shape[3];

    for (int b = 0; b < B; b++) {
      for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
          for (int k = -d; k < d + 1; k++) {
            for (int l = -d; l < d + 1; l++) {
              int x1_index = i + pad_size_;
              int y1_index = j + pad_size_;
              int x2_index = x1_index + k;
              int y2_index = y1_index + l;

              float sum = 0.f;
              for (int c = 0; c < ic; c++) {
                for (int h = 0; h < K; h++) {
                  for (int w = 0; w < K; w++) {
                    int idx1 = b * ic * ih * iw + c * ih * iw +
                               (x1_index + h) * iw + (y1_index + w);
                    int idx2 = b * ic * ih * iw + c * ih * iw +
                               (x2_index + h) * iw + (y2_index + w);
                    sum += x1_pad_data[idx1] * x2_pad_data[idx2];
                  }
                }
              }
              int out_idx = b * oc * oh * ow + (l + d + D * (k + d)) * oh * ow +
                            i * ow + j;
              out_data[out_idx] = sum / static_cast<float>(ic * K * K);
            }
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("correlation");
    op_desc->SetInput("Input1", {input1_});
    op_desc->SetInput("Input2", {input2_});
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("pad_size", pad_size_);
    op_desc->SetAttr("kernel_size", kernel_size_);
    op_desc->SetAttr("max_displacement", max_displacement_);
    op_desc->SetAttr("stride1", stride1_);
    op_desc->SetAttr("stride2", stride2_);
    op_desc->SetAttr("stride2", stride2_);
    op_desc->SetAttr("corr_type_multiply", corr_type_multiply_);
  }

  void PrepareData() override {
    std::vector<float> x1_data(x_dims_.production());
    fill_data_rand(x1_data.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(input1_, x_dims_, x1_data.data());

    std::vector<float> x2_data(x_dims_.production());
    fill_data_rand(x2_data.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(input2_, x_dims_, x2_data.data());
  }
};

TEST(correlation, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  std::unique_ptr<arena::TestCase> tester(
      new CorrelationComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

}  // namespace lite
}  // namespace paddle
