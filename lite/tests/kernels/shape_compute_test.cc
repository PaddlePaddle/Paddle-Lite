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
class ShapeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "Input";
  std::string out_ = "Out";
  DDim dims_;

 public:
  ShapeComputeTester(const Place& place, const std::string& alias, DDim dims)
      : TestCase(place, alias), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* input = scope->FindTensor(input_);
    CHECK(input);
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    int64_t sz = input->dims().size();
    out->Resize(DDim({sz}));
    auto* out_data = out->mutable_data<int>();
    for (int i = 0; i < input->dims().size(); ++i) {
      out_data[i] = input->dims()[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("shape");
    op_desc->SetInput("Input", {input_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

void TestShapeHelper(Place place,
                     float abs_error,
                     std::vector<int64_t> x_dims) {
  std::unique_ptr<arena::TestCase> tester(
      new ShapeComputeTester(place, "def", DDim(x_dims)));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void test_shape(Place place, float abs_error) {
  TestShapeHelper(place, abs_error, {2, 3, 4, 5});
  TestShapeHelper(place, abs_error, {3, 4, 5});
  TestShapeHelper(place, abs_error, {4, 5});
  TestShapeHelper(place, abs_error, {5});
}

TEST(shape, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  test_shape(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
