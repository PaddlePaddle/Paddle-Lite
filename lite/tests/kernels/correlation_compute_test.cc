// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class CorrelationComputeTester : public arena::TestCase {
 protected:
  std::string input1_ = "x1";
  std::string input2_ = "x2";
  std::string output_ = "out";
  DDim x_dims_;

 public:
  CorrelationComputeTester(const Place& place,
                           const std::string& alias,
                           const DDim& x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    // out->Resize(x_dims_);
    // auto* x = scope->FindTensor(x_);
    // const float* x_data = x->data<float>();
    // float* out_data = out->mutable_data<float>();
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("correlation");
    op_desc->SetInput("Input1", {input1_});
    op_desc->SetInput("Input2", {input2_});
    op_desc->SetOutput("Output", {output_});
  }

  void PrepareData() override {
    // std::vector<float> x_data(x_dims_.production());
    // fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
    // SetCommonTensor(x_, x_dims_, x_data.data());
  }
};

TEST(correlation, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
}

}  // namespace lite
}  // namespace paddle
