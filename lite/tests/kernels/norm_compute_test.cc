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

class NormComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  int axis_ = 1;
  float epsilon_ = 1e-9;
  DDim dims_{{3, 5, 4, 4}};
  bool bias_after_scale_;

 public:
  NormComputeTester(const Place& place,
                    const std::string& alias,
                    int axis,
                    float epsilon,
                    DDim dims)
      : TestCase(place, alias), axis_(axis), epsilon_(epsilon), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    int axis = axis_ < 0 ? axis + dims_.size() : axis_;
    int pre_n = dims_.count(0, axis);
    int n = dims_[axis];
    int post_n = dims_.count(axis + 1, dims_.size());
    for (int i = 0; i < pre_n; i++) {
      for (int k = 0; k < post_n; k++) {
        float sum = epsilon_;
        const float* in_tmp = x_data + i * n * post_n + k;
        for (int j = 0; j < n; j++) {
          sum += in_tmp[j * post_n] * in_tmp[j * post_n];
        }
        sum = std::sqrt(sum);
        float* out_tmp = out_data + i * n * post_n + k;
        for (int j = 0; j < n; j++) {
          out_tmp[j * post_n] = in_tmp[j * post_n] / sum;
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("norm");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("epsilon", epsilon_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};
void test_norm(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int axis : {1}) {
    for (float epsilon : {1e-9}) {
      std::unique_ptr<arena::TestCase> tester(
          new NormComputeTester(place, "def", axis, epsilon, dims));
      arena::Arena arena(std::move(tester), place, 2e-5);
      arena.TestPrecision();
    }
  }
}

TEST(Norm, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_norm(place);
#endif
}

}  // namespace lite
}  // namespace paddle
