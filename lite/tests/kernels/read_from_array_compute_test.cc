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

class ReadFromArrayComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_0 = "in_0";
  std::string input_1 = "in_1";
  std::string input_2 = "in_2";
  std::string input_i = "i";
  std::string output = "out";
  DDim dims_{{3, 5, 4, 4}};
  int i_;

 public:
  ReadFromArrayComputeTester(const Place& place,
                             const std::string& alias,
                             const int i,
                             DDim dims)
      : TestCase(place, alias), i_(i), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output);
    CHECK(out);
    auto* in_0 = scope->FindTensor(input_0);
    auto* in_1 = scope->FindTensor(input_1);
    auto* in_2 = scope->FindTensor(input_2);
    auto* id_tensor = scope->FindTensor(input_i);
    std::vector<const TensorLite*> in_vec = {in_0, in_1, in_2};
    int cur_in_num = in_vec.size();

    int id = id_tensor->data<int>()[0];
    out->Resize(dims_);
    const auto* in_data = in_vec[id]->data<float>();
    auto* o_data = out->mutable_data<float>();
    int n = in_vec[id]->numel();
    memcpy(o_data, in_data, sizeof(float) * n);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("read_from_array");
    op_desc->SetInput("X", {input_0, input_1, input_2});
    op_desc->SetInput("I", {input_i});
    op_desc->SetOutput("Out", {output});
  }

  void PrepareData() override {
    std::vector<std::string> in_vec = {input_0, input_1, input_2};
    for (auto in : in_vec) {
      std::vector<float> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        data[i] = std::rand() * 1.0f / RAND_MAX;
      }
      SetCommonTensor(in, dims_, data.data());
    }

    DDimLite dims_i{{1}};
    int a = 1;
    SetCommonTensor(input_i, dims_i, &a);
  }
};

void test_read_from_array(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int i : {1, 2}) {
    std::unique_ptr<arena::TestCase> tester(
        new ReadFromArrayComputeTester(place, "def", i, dims));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

TEST(ReadFromArray, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_read_from_array(place);
#endif
}

}  // namespace lite
}  // namespace paddle
