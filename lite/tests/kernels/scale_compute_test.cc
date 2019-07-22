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

class ScaleComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  float scale_ = 0.;
  float bias_ = 0.;
  DDim dims_{{100, 20}};
  bool bias_after_scale_;

 public:
  ScaleComputeTester(const Place& place,
                     const std::string& alias,
                     float scale,
                     float bias,
                     bool bias_after_scale)
      : TestCase(place, alias),
        scale_(scale),
        bias_(bias),
        bias_after_scale_(bias_after_scale) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    float bias = bias_;

    if (!bias_after_scale_) {
      bias *= scale_;
    }

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] * scale_ + bias;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("scale");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("scale", scale_);
    op_desc->SetAttr("bias", bias_);
    op_desc->SetAttr("bias_after_scale", bias_after_scale_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

TEST(Scale, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

  for (float scale : {0.123, 2., -1.2}) {
    for (float bias : {1., 0., -1.2331}) {
      for (bool bias_before : {true, false}) {
        std::unique_ptr<arena::TestCase> tester(
            new ScaleComputeTester(place, "def", scale, bias, bias_before));
        arena::Arena arena(std::move(tester), place, 2e-5);
        arena.TestPrecision();
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
