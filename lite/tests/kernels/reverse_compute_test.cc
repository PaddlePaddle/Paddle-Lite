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

class ReverseComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::string alias_ = "fp32";
  std::vector<int> axis_;
  DDim dims_{{2, 5, 20, 30}};

 public:
  ReverseComputeTester(const Place& place,
                       const std::string& alias,
                       std::vector<int> axis,
                       DDim dims)
      : TestCase(place, alias), alias_(alias), axis_(axis), dims_(dims) {}

  template <typename indtype>
  void ReverseAB(indtype* x, int size, int a, int b) {
    for (int i = 0; i < size; i += a) {
      for (int j = 0; j < (a / b) / 2; j++) {
        for (int k = 0; k < b; k++) {
          indtype temp;
          temp = x[i + (a / b - 1 - j) * b + k];
          x[i + (a / b - 1 - j) * b + k] = x[i + j * b + k];
          x[i + j * b + k] = temp;
        }
      }
    }
  }
  template <typename indtype>
  void RunBaselineKernel(Scope* scope) {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* output_data = out->mutable_data<indtype>();
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<indtype>();

    for (int i = 0; i < dims_.count(0, dims_.size()); i++)
      output_data[i] = x_data[i];

    for (int i = 0; i < axis_.size(); i++) {
      int a = dims_.count(axis_[i], dims_.size());
      int b = dims_.count(axis_[i] + 1, dims_.size());
      ReverseAB(output_data, dims_.count(0, dims_.size()), a, b);
    }
  }

  void RunBaseline(Scope* scope) override {
    if (alias_ == "fp32") {
      RunBaselineKernel<float>(scope);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("reverse");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    if (alias_ == "fp32") {
      std::vector<float> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
      }
      SetCommonTensor(input_, dims_, data.data());
    }
  }
};

void TestReverse(const Place& place) {
  std::vector<std::vector<int>> Axis = {{
                                            0,
                                        },
                                        {
                                            1,
                                        },
                                        {
                                            2,
                                        },
                                        {1, 3}};
  for (std::vector<int>& axis : Axis) {
    for (int n : {1, 3}) {
      for (int c : {3, 6}) {
        for (int h : {9, 18}) {
          for (int w : {9, 18}) {
            std::vector<std::string> alias_vec{"fp32"};
            for (std::string alias : alias_vec) {
              DDim dims{{n, c, h, w}};
              std::unique_ptr<arena::TestCase> tester(
                  new ReverseComputeTester(place, alias, axis, dims));
              arena::Arena arena(std::move(tester), place, 2e-5);
              arena.TestPrecision();
            }
          }
        }
      }
    }
  }
}

TEST(Reverse, precision) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  TestReverse(place);
}

}  // namespace lite
}  // namespace paddle
