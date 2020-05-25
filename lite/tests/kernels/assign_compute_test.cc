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

class AssignComputeTester : public arena::TestCase {
 protected:
  std::string input_ = "X";
  std::string output_ = "Out";
  DDim dims_{{100, 20}};

 public:
  AssignComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("assign");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

void TestAssign(const Place& place) {
  std::unique_ptr<arena::TestCase> tester(
      new AssignComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

TEST(Assign, precision) {
  Place place;
#ifdef LITE_WITH_ARM
  place = TARGET(kHost);
#else
  return;
#endif

  TestAssign(place);
}

}  // namespace lite
}  // namespace paddle
