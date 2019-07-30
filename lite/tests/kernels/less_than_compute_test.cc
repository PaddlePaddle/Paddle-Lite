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

class LessThanTester : public arena::TestCase {
 protected:
  std::string input_x_ = "x";
  std::string input_y_ = "y";
  std::string output_ = "out";
  int axis_ = 1;
  bool force_cpu_ = 0;
  DDim dims_{{3, 5, 4, 4}};

 public:
  LessThanTester(const Place& place,
                 const std::string& alias,
                 bool force_cpu,
                 int axis,
                 DDim dims)
      : TestCase(place, alias),
        axis_(axis),
        force_cpu_(force_cpu),
        dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<bool>();
    auto* x = scope->FindTensor(input_x_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(input_y_);
    const auto* y_data = y->data<float>();
    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] < y_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("less_than");
    op_desc->SetInput("X", {input_x_});
    op_desc->SetInput("Y", {input_y_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("force_cpu", force_cpu_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());
    std::vector<float> datay(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      data[i] =  1.1;
      datay[i] = 1.1;
    }

    SetCommonTensor(input_x_, dims_, data.data());
    SetCommonTensor(input_y_, dims_, datay.data());
  }
};
void test_lessthan(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int axis : {1}) {
    for (bool force_cpu : {0}) {
      std::unique_ptr<arena::TestCase> tester(
          new LessThanTester(place, "def", force_cpu, axis, dims));
      arena::Arena arena(std::move(tester), place, 1);
      arena.TestPrecision();
    }
  }
}
TEST(LessThan, precision) {
// #ifdef LITE_WITH_X86
// //   Place place(TARGET(kX86));
// // #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_lessthan(place);
#endif
}

}  // namespace lite
}  // namespace paddle
