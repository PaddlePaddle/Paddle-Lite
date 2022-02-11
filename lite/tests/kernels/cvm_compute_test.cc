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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class CvmComputeTester : public arena::TestCase {
 protected:
  std::string input_ = "X";
  std::string output_ = "Y";
  bool use_cvm_ = false;
  DDim x_dims_{{2, 5}};

 public:
  CvmComputeTester(const Place& place,
                   const std::string& alias,
                   bool use_cvm,
                   DDim x_dims)
      : TestCase(place, alias), use_cvm_(use_cvm), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    std::vector<int64_t> output_shape;
    if (use_cvm_)
      output_shape = {x_dims_[0], x_dims_[1]};
    else
      output_shape = {x_dims_[0], x_dims_[1] - 2};
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    int n = x_dims_[0];
    int c = x_dims_[1];

    if (use_cvm_) {
      for (int i = 0; i < n; i++) {
        int cursor = i * c;
        output_data[cursor] = ::log(x_data[cursor] + 1);
        output_data[cursor + 1] =
            ::log(x_data[cursor + 1] + 1) - output_data[cursor];
        for (int j = 2; j < c; j++) {
          output_data[cursor + j] = x_data[cursor + j];
        }
      }
    } else {
      for (int i = 0; i < n; i++) {
        std::memcpy(output_data + i * (c - 2),
                    x_data + i * c + 2,
                    (c - 2) * sizeof(float));
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("cvm");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Y", {output_});

    op_desc->SetAttr("use_cvm", use_cvm_);
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(input_, x_dims_, x_data.data());
  }
};

void TestCvm(const Place& place) {
  for (auto x_shape : std::vector<std::vector<int64_t>>{
           {1, 3}, {32, 20}, {200, 200}, {2000, 4000}}) {
    for (auto use_cvm : std::vector<bool>{true, false}) {
      std::unique_ptr<arena::TestCase> tester(
          new CvmComputeTester(place, "def", use_cvm, DDim(x_shape)));
      arena::Arena arena(std::move(tester), place, 2e-5);
      arena.TestPrecision();
    }
  }
}

TEST(Cvm, precision) {
  LOG(INFO) << "test cvm op";
  Place place;
#if defined(LITE_WITH_XPU) && !defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif
  TestCvm(place);
}

}  // namespace lite
}  // namespace paddle
