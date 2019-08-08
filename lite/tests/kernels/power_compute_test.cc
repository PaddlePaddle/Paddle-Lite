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

class PowerComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  float scale_ = 0.;
  float shift_ = 0.;
  float power_ = 0.;
  DDim dims_{{5, 2}};

 public:
  PowerComputeTester(const Place& place,
                     const std::string& alias,
                     float scale,
                     float shift,
                     float power)
      : TestCase(place, alias), scale_(scale), shift_(shift), power_(power) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = std::pow((x_data[i] * scale_ + shift_), power_);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("power");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("scale", scale_);
    op_desc->SetAttr("shift", shift_);
    op_desc->SetAttr("power", power_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.01;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

void test_power(Place place) {
  for (float scale : {0.923, 2., 1.2}) {
    for (float shift : {1., 0., 1.2331}) {
      for (float power : {1., 1.2, 1.6}) {
        std::unique_ptr<arena::TestCase> tester(
            new PowerComputeTester(place, "def", scale, shift, power));
        arena::Arena arena(std::move(tester), place, 2e-4);
        arena.TestPrecision();
      }
    }
  }
}

TEST(Power, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_power(place);
#endif
}

}  // namespace lite
}  // namespace paddle
