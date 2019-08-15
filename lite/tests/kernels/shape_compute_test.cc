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
class ShapeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "Input";
  std::string out_ = "Out";
  DDim dims_;

 public:
  ShapeComputeTester(const Place& place, const std::string& alias, DDim dims)
      : TestCase(place, alias), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* input = scope->FindTensor(x_);
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
    op_desc->SetInput("Input", {x_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> in_data(dims_.production());
    for (int i = 0; i < dims_.production(); ++i) {
      in_data[i] = i;
    }
    SetCommonTensor(x_, dims_, in_data.data());
  }
};

void test_shape(Place place) {
  for (int N : {1, 2, 3, 4}) {
    for (int C : {1, 2, 3, 4}) {
      for (int H : {1, 2, 3, 4}) {
        for (int W : {1, 2, 3, 4}) {
          std::unique_ptr<arena::TestCase> tester(
              new ShapeComputeTester(place, "def", DDim({N, C, H, W})));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(shape, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_shape(place);
#endif
}

}  // namespace lite
}  // namespace paddle
