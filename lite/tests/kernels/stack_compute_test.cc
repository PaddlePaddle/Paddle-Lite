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

void stack(std::vector<const lite::Tensor*> x, lite::Tensor* y, int axis) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int n = x.size();
  auto* y_data = y->mutable_data<float>();
  std::vector<const float*> x_datas(n);
  for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<float>();

  int pre = 1, post = 1;
  auto& dim = x[0]->dims();
  for (auto i = 0; i < axis; ++i) pre *= dim[i];
  for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

  auto x_data_arr = x_datas.data();

  size_t x_offset = 0;
  size_t y_offset = 0;
  for (int i = 0; i < pre; i++) {
    for (int j = 0; j < n; j++) {
      std::memcpy(
          y_data + y_offset, x_data_arr[j] + x_offset, post * sizeof(float));
      y_offset += post;
    }
    x_offset += post;
  }
}

class StackComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input1_ = "X1";
  std::string input2_ = "X2";
  std::string output_ = "Out";
  int axis_ = 0;
  DDim dims_{{1, 5, 6, 7}};

 public:
  StackComputeTester(const Place& place, const std::string& alias, float axis)
      : TestCase(place, alias), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    std::vector<const lite::Tensor*> x;
    x.emplace_back(scope->FindTensor(input1_));
    x.emplace_back(scope->FindTensor(input2_));
    auto input_dims = x[0]->dims();
    int rank = input_dims.size();
    if (axis_ < 0) axis_ += (rank + 1);
    auto vec = input_dims.Vectorize();
    vec.insert(vec.begin() + axis_, x.size());
    out->Resize(vec);
    stack(x, out, axis_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("stack");
    op_desc->SetInput("X", {input1_, input2_});
    op_desc->SetOutput("Y", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.01;
    }

    SetCommonTensor(input1_, dims_, data.data());
    SetCommonTensor(input2_, dims_, data.data());
  }
};

void test_stack(Place place) {
  for (float axis : {0, 1, 3}) {
    std::unique_ptr<arena::TestCase> tester(
        new StackComputeTester(place, "def", axis));
    arena::Arena arena(std::move(tester), place, 2e-4);
    arena.TestPrecision();
  }
}

TEST(Stack, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_stack(place);
#endif
}

}  // namespace lite
}  // namespace paddle
