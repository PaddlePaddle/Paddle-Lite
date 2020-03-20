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
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -5.f, 5.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

void test_increment(Place place, float abs_error) {
  std::vector<std::vector<int64_t>> x_dims{{3, 5, 4, 4}, {3, 5}, {1}};
  for (auto dims : x_dims) {
    for (float step : {1, 2}) {
#if LITE_WITH_NPU
      if (dims.size() != 1) continue;
#endif
      std::unique_ptr<arena::TestCase> tester(
          new IncrementComputeTester(place, "def", step, DDim(dims)));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(Increment, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_increment(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
