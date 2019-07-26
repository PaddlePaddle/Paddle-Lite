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
#include <stdio.h>
#include <stdlib.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

class WriteToArrayComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_0 = "x";
  std::string input_1 = "i";
  std::string output_0 = "out0";
  std::string output_1 = "out1";
  std::string output_2 = "out2";
  DDim dims_{{3, 5, 4, 4}};
  int i_;

 public:
  WriteToArrayComputeTester(const Place& place,
                            const std::string& alias,
                            const int i,
                            DDim dims)
      : TestCase(place, alias), i_(i), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out_0 = scope->NewTensor(output_0);
    auto* out_1 = scope->NewTensor(output_1);
    auto* out_2 = scope->NewTensor(output_2);
    CHECK(out_0);
    CHECK(out_1);
    CHECK(out_2);
    std::vector<TensorLite*> out_vec = {out_0, out_1, out_2};

    auto* x = scope->FindTensor(input_0);
    const auto* x_data = x->data<float>();
    auto* id = scope->FindTensor(input_1);
    const auto* id_data = id->data<float>();
    int n = x->numel();
    int cur_out_num = out_vec.size();
    for (int i = cur_out_num; i < id_data[0] + 1; i++) {
      char buffer[30];
      snprintf(buffer, sizeof(buffer), "out%d", i);
      auto out = scope->NewTensor(buffer);
      out_vec.push_back(out);
    }
    out_vec[id_data[0]]->Resize(dims_);
    auto* out_data = out_vec[id_data[0]]->mutable_data<float>();
    memcpy(out_data, x_data, sizeof(float) * n);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("write_to_array");
    op_desc->SetInput("X", {input_0});
    op_desc->SetInput("I", {input_1});
    op_desc->SetOutput("Out", {output_0, output_1, output_2});
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_0, dims_, data.data());

    std::vector<int> data_1(1);
    data_1[0] = i_;
    DDimLite dims_2{{1}};
    SetCommonTensor(input_1, dims_2, data_1.data());

    SetCommonTensor(output_0, dims_2, data_1.data());
    SetCommonTensor(output_1, dims_2, data_1.data());
    SetCommonTensor(output_2, dims_2, data_1.data());
  }
};
void test_write_to_array(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int i : {1, 4}) {
    std::unique_ptr<arena::TestCase> tester(
        new WriteToArrayComputeTester(place, "def", i, dims));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

TEST(WriteToArray, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_write_to_array(place);
#endif
}

}  // namespace lite
}  // namespace paddle
