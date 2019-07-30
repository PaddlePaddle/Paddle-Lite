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

class IncrementComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  float step_ = 1.f;
  DDim dims_{{3, 5, 4, 4}};
  bool bias_after_scale_;

 public:
  IncrementComputeTester(const Place& place,
                         const std::string& alias,
                         float step,
                         DDim dims)
      : TestCase(place, alias), step_(step), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    int n = dims_.production();
    for (int i = 0; i < n; i++) {
      out_data[i] = x_data[i] + step_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("increment");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("step", step_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};
void test_increment(Place place) {
  DDimLite dims_0{{3, 5, 4, 4}};
  DDimLite dims_1{{3, 5}};
  for (auto dims : {dims_0, dims_1}) {
    for (float step : {1, 2}) {
      std::unique_ptr<arena::TestCase> tester(
          new IncrementComputeTester(place, "def", step, dims));
      arena::Arena arena(std::move(tester), place, 2e-5);
      arena.TestPrecision();
    }
  }
}

TEST(Increment, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_increment(place);
#endif
}

}  // namespace lite
}  // namespace paddle
