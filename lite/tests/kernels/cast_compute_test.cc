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
  int in_dtype_;
  int out_dtype_;
  DDim x_dims_{{2, 2}};

 public:
  CastComputeTester(const Place& place,
                    const std::string& alias,
                    int in_dtype,
                    int out_dtype)
      : TestCase(place, alias), in_dtype_(in_dtype), out_dtype_(out_dtype) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(x_dims_);

    if (out_dtype_ == 5 && in_dtype_ == 20) {
      auto* x = scope->FindTensor(input_);
      auto* x_data = x->data<unsigned char>();
      auto* output_data = out->mutable_data<float>();
      for (int i = 0; i < x_dims_.production(); i++) {
        *output_data = static_cast<float>(*x_data);
        output_data++;
        x_data++;
      }
    } else if (out_dtype_ == 5 && in_dtype_ == 21) {
      auto* output_data = out->mutable_data<float>();
      auto* x = scope->FindTensor(input_);
      auto* x_data = x->data<char>();
      for (int i = 0; i < x_dims_.production(); i++) {
        *output_data = static_cast<float>(*x_data);
        output_data++;
        x_data++;
      }
    } else if (out_dtype_ == 5 && in_dtype_ == 2) {
      auto* output_data = out->mutable_data<float>();
      auto* x = scope->FindTensor(input_);
      auto* x_data = x->data<int32_t>();
      for (int i = 0; i < x_dims_.production(); i++) {
        *output_data = static_cast<float>(*x_data);
        output_data++;
        x_data++;
      }
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
    if (in_dtype_ == 20) {
      std::vector<unsigned char> x_data(x_dims_.production());
      for (int i = 0; i < x_dims_.production(); i++) {
        x_data[i] = static_cast<unsigned char>(i % 128);
      }
      SetCommonTensor(input_, x_dims_, x_data.data());
    } else if (in_dtype_ == 21) {
      std::vector<char> x_data(x_dims_.production());
      for (int i = 0; i < x_dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        x_data[i] = sign * static_cast<char>(i % 128);
      }
      SetCommonTensor(input_, x_dims_, x_data.data());
    } else if (in_dtype_ == 2) {
      std::vector<int32_t> x_data(x_dims_.production());
      for (int i = 0; i < x_dims_.production(); i++) {
        int sign = i % 3 == 0 ? -1 : 1;
        x_data[i] = sign * static_cast<int32_t>(i % 128);
      }
      SetCommonTensor(input_, x_dims_, x_data.data());
    } else {
      LOG(FATAL) << "not implemented!";
    }
  }
};

TEST(Cast, precision) {
  LOG(INFO) << "test cast op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  std::unique_ptr<arena::TestCase> tester(
      new CastComputeTester(place, "def", 20, 5));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();

// std::unique_ptr<arena::TestCase> tester1(
//    new CastComputeTester(place, "def", 2, 5));
// arena::Arena arena1(std::move(tester1), place, 2e-5);
// arena1.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
