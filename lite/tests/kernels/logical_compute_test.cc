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

bool _logical_xor_func(const bool& a, const bool& b) {
  return (a || b) && !(a && b);
}
bool _logical_and_func(const bool& a, const bool& b) { return (a && b); }
template <bool (*T)(const bool&, const bool&)>
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
      // out_data[i] = (x_data[i] || y_data[i]) && !((x_data[i] && y_data[i]));
      out_data[i] = T(x_data[i], y_data[i]);
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
    bool* data;
    bool* datay;
    data = reinterpret_cast<bool*>(malloc(dims_.production() * sizeof(bool)));
    datay = reinterpret_cast<bool*>(malloc(dims_.production() * sizeof(bool)));
    LOG(INFO) << "dims_.production()"
              << ":::" << dims_.production();

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = 1;
      datay[i] = 1;
    }

    SetCommonTensor(input_x_, dims_, data);
    SetCommonTensor(input_y_, dims_, datay);
  }
};

void test_logical(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  std::unique_ptr<arena::TestCase> logical_xor_tester(
      new LogicalXorTester<_logical_xor_func>(place, "def", dims));
  arena::Arena arena_xor(std::move(logical_xor_tester), place, 1);

  arena_xor.TestPrecision();

  std::unique_ptr<arena::TestCase> logical_and_tester(
      new LogicalXorTester<_logical_and_func>(place, "def", dims));
  arena::Arena arena_and(std::move(logical_and_tester), place, 1);

  arena_and.TestPrecision();
}
TEST(Logical, precision) {
// #ifdef LITE_WITH_X86
// //   Place place(TARGET(kX86));
// // #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_logical(place);
#endif
}

}  // namespace lite
}  // namespace paddle
