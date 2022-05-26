// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
class IsEmptyComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  DDim x_dims_;

 public:
  IsEmptyComputeTester(const Place& place,
                       const std::string& alias,
                       DDim x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* x = scope->FindTensor(x_);
    auto* out = scope->NewTensor(out_);

    out->Resize(DDim({1}));
    auto* out_data = out->mutable_data<bool>();
    out_data[0] = (x->numel() == 0) ? true : false;
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("is_empty");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> din(x_dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, din.data());
  }
};

void TestIsEmptyHelper(Place place,
                       float abs_error,
                       std::vector<int64_t> x_dims) {
  std::unique_ptr<arena::TestCase> tester(
      new IsEmptyComputeTester(place, "def", DDim(x_dims)));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestIsEmpty(Place place, float abs_error) {
  TestIsEmptyHelper(place, abs_error, {2, 3, 4, 5});
#if !defined(LITE_WITH_XPU)
  TestIsEmptyHelper(place, abs_error, {0});
#endif
}

TEST(is_empty, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#else
  return;
#endif

  TestIsEmpty(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
