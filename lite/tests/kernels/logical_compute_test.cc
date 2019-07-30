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

class LogicalXorTester : public arena::TestCase {
 protected:
  std::string input_x_ = "x";
  std::string input_y_ = "y";
  std::string output_ = "out";
  DDim dims_{{3, 5, 4, 4}};

 public:
  LogicalXorTester(const Place& place, const std::string& alias, DDim dims)
      : TestCase(place, alias), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    bool* out_data = out->mutable_data<bool>();
    auto* x = scope->FindTensor(input_x_);
    const bool* x_data = x->data<bool>();
    auto* y = scope->FindTensor(input_y_);
    const bool* y_data = y->data<bool>();
    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = (x_data[i] || y_data[i]) && !((x_data[i] && y_data[i]));

    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("logical_xor");
    op_desc->SetInput("X", {input_x_});
    op_desc->SetInput("Y", {input_y_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {

    // std::vector<bool> data(dims_.production());
    // std::vector<char> datay(dims_.production());
    bool data[dims_.production()];
    bool datay[dims_.production()];
        LOG(INFO) << "dims_.production()" <<":::" << dims_.production();


    for (int i = 0; i < dims_.production(); i++) {
      data[i] =  1;
      datay[i] = 1;
    }

    SetCommonTensor(input_x_, dims_, &data[0]);
    SetCommonTensor(input_y_, dims_, &datay[0]);

  }
};
void test_logicalxor(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  std::unique_ptr<arena::TestCase> tester(
      new LogicalXorTester(place, "def", dims));
  arena::Arena arena(std::move(tester), place, 1);

  arena.TestPrecision();
}
TEST(LessThan, precision) {
// #ifdef LITE_WITH_X86
// //   Place place(TARGET(kX86));
// // #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_logicalxor(place);
#endif
}

}  // namespace lite
}  // namespace paddle
