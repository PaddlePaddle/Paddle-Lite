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

#include "lite/core/test/arena/framework.h"
#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"

namespace paddle {
namespace lite {

class ScaleComputeTester : public arena::TestCase {
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  float scale_ = 1.2f;
  float bias_ = 0.f;
  DDim dims_{{3, 2, 10}};

 public:
  explicit ScaleComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] * scale_ + bias_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("scale");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("scale", scale_);
    op_desc->SetAttr("bias", bias_);
    op_desc->SetAttr("bias_after_scale", false);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

TEST(scale, basic) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif
  std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place);

  arena.TestPrecision();
}

}  // namespace lite
}  // namespace paddle
