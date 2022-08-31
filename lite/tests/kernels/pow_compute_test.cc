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

namespace paddle {
namespace lite {

class PowComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  float factor_ = 0.;
  DDim dims_{{5, 8}};

 public:
  PowComputeTester(const Place& place, const std::string& alias, float factor)
      : TestCase(place, alias), factor_(factor) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = std::pow(x_data[i], factor_);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("pow");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("factor", factor_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = (i + 1) * 0.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

void test_pow(Place place, float abs_error) {
  for (float factor : {1., 1.2, 1.6}) {
    std::unique_ptr<arena::TestCase> tester(
        new PowComputeTester(place, "def", factor));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Pow, precision) {
  float abs_error = 2e-4;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#else
  return;
#endif
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 3e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif
  test_pow(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
