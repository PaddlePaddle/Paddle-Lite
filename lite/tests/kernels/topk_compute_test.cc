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

class TopkComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string out_val_ = "out_val";
  std::string out_ind_ = "out_ind";
  int K_ = 1;
  DDim dims_{{3, 5, 4, 4}};

 public:
  TopkComputeTester(const Place& place,
                    const std::string& alias,
                    int K,
                    DDim dims)
      : TestCase(place, alias), K_(K), , dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
  }
  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("topk");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {out_val_, out_ind_});
    op_desc->SetAttr("K", K_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};
void test_topk(Place place) {
  DDimLite dims_0{{3, 5, 4, 4}};
  DDimLite dims_1{{8}};
  for (int K : {1, 2}) {
    for (auto dims : {dims_0, dims_1})
      std::unique_ptr<arena::TestCase> tester(
          new TopkComputeTester(place, "def", K, dims));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

TEST(Topk, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_topk(place);
#endif
}

}  // namespace lite
}  // namespace paddle
