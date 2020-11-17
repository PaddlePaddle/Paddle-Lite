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
#include <cmath>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class SinComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "sin";
  DDim x_dims_{{1, 2, 3, 4}};
  std::string x_ = "x";
  std::string out_ = "out";

 public:
  SinComputeTest(const Place& place, const std::string& alias, DDim x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(x_dims_);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();

    for (int i = 0; i < x_dims_.production(); i++) {
      out_data[i] = sin(x_data[i]);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());
  }
};

void test_sin(Place place) {
  float abs_error = 2e-4;
  for (auto x_dims : std::vector<std::vector<int64_t>>{
           {1, 2, 3, 4}, {2, 3, 4}, {3, 4}, {4}}) {
    std::unique_ptr<arena::TestCase> tester(
        new SinComputeTest(place, "def", DDim(x_dims)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Sin, precision) {
#ifdef LITE_WITH_ARM
  Place place(TARGET(kHost));
  test_sin(place);
#endif
}

}  // namespace lite
}  // namespace paddle
