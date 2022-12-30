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
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

class HxpdyComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "hxpdy";
  std::string x_ = "X";
  std::string y_ = "Y";
  std::string output_ = "Out";
  DDim dims_{{2, 5, 20, 30}};

 public:
  HxpdyComputeTester(
      const Place& place, const std::string& alias, const std::string& op_type)
      : TestCase(place, alias), op_type_(op_type) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* y = scope->FindTensor(y_);
    const auto* x_data = x->data<float>();
    const auto* y_data = y->data<float>();
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* output_data = out->mutable_data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      output_data[i] = 0.5 * x_data[i] + 2 * y_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("hxpdy");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<float> x_data(dims_.production());
    std::vector<float> y_data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
      y_data[i] = sign * static_cast<float>(i % 128) * 0.017f - 0.011;
    }
    SetCommonTensor(x_, dims_, x_data.data());
    SetCommonTensor(y_, dims_, y_data.data());
  }
};

TEST(Hxpdy, precision) {
  LOG(INFO) << "test hxpdy op";
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  std::unique_ptr<arena::TestCase> tester(
      new HxpdyComputeTester(place, "def","hxpdy"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
