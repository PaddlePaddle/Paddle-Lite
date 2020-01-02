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

class ScaleComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  DDim x_dims_{{100, 20}};
  float scale_ = 0.;
  float bias_ = 0.;
  bool bias_after_scale_;

 public:
  ScaleComputeTester(const Place& place,
                     const std::string& alias,
                     const DDim& x_dims,
                     float scale,
                     float bias,
                     bool bias_after_scale)
      : TestCase(place, alias),
        x_dims_(x_dims),
        scale_(scale),
        bias_(bias),
        bias_after_scale_(bias_after_scale) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(x_dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->data<float>();

    float bias = bias_;

    if (!bias_after_scale_) {
      bias *= scale_;
    }

    for (int i = 0; i < x_dims_.production(); i++) {
      out_data[i] = x_data[i] * scale_ + bias;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("scale");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("scale", scale_);
    op_desc->SetAttr("bias", bias_);
    op_desc->SetAttr("bias_after_scale", bias_after_scale_);
  }

  void PrepareData() override {
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());
  }
};

TEST(Scale, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 4e-3;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
  abs_error = 3e-4;  // Some operations use fp16 in XPU
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto x_dims :
       std::vector<std::vector<int64_t>>{{5, 2, 3, 4}, {8, 3, 5}, {12, 3}}) {
    for (float scale : {0.123, 2., -1.2}) {
      for (float bias : {1., 0., -1.2331}) {
        for (bool bias_after_scale : {true, false}) {
          std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester(
              place, "def", DDim(x_dims), scale, bias, bias_after_scale));
          arena::Arena arena(std::move(tester), place, abs_error);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(Scale, performance) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester(
      place, "def", DDim(std::vector<int64_t>{5, 2, 3, 4}), 1.2, 1.1, true));

  // To modify the arm context, one can retrive the context as follows.
  // #ifdef LITE_WITH_ARM
  //   tester->context()->As<ARMContext>();
  // #endif

  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPerformance(100);
}

}  // namespace lite
}  // namespace paddle
