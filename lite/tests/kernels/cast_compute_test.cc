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

class CastComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  int in_dtype_ = 21;
  int out_dtype_ = 5;
  DDim x_dims_{{2, 2, 2, 2}};

 public:
  CastComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(x_dims_);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<char>();

    int num = x_dims_[0];
    int channel = x_dims_[1];
    int size = x_dims_[2] * x_dims_[3];
    int in_channel = channel * size;

    auto* output_data_tmp = output_data;
    auto* x_data_tmp = x_data;
    for (int i = 0; i < x_dims_.production(); i++) {
      *output_data_tmp = static_cast<float>(*x_data_tmp);
      output_data_tmp++;
      x_data_tmp++;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("cast");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("in_dtype", in_dtype_);
    op_desc->SetAttr("out_dtype", out_dtype_);
  }

  void PrepareData() override {
    std::vector<char> x_data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<char>(i % 128);
    }
    SetCommonTensor(input_, x_dims_, x_data.data());
  }
};

TEST(Cast, precision) {
  LOG(INFO) << "test cast op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  std::unique_ptr<arena::TestCase> tester(new CastComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
