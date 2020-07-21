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

class AffineGridComputeTester : public arena::TestCase {
 protected:
  std::string input_ = "Theta";
  std::string output_ = "Output";
  std::vector<int> output_shape{{-1, 1, 2, 3}};
  DDim x_dims_{{1, 2, 3}};
  DDim o_dims_{{1, 5, 20, 2}};

 public:
  AffineGridComputeTester(const Place& place,
                             const std::string& alias,
                             int n,
                             int c,
                             int h,
                             int w)
      : TestCase(place, alias) {
    x_dims_ = DDim(std::vector<int64_t>({n, 2, 3}));
    o_dims_ = DDim(std::vector<int64_t>({n, h, w, 2}));
    output_shape = std::vector<int>({-1, n, h, w});
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(o_dims_);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    int num = x_dims_[0];

  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("affine_grid");
    op_desc->SetInput("Theta", {input_});
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("output_shape", output_shape);

  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      x_data[i] = 1;

    }

    SetCommonTensor(input_, x_dims_, x_data.data());

  }
};

TEST(AffineGrid, precision) {
  LOG(INFO) << "test affine_grid op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (int n : {17}) {
    for (int c : { 2}) {
      for (int h : {5}) {
        for (int w : {7}) {

            std::unique_ptr<arena::TestCase> tester(
                new AffineGridComputeTester(
                    place, "def", n, c, h, w));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
