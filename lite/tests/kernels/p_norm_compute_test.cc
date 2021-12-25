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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

void p_norm_compute(const float* input,
                    const int pre_n,
                    const int n,
                    const int post_n,
                    const float epsilon,
                    float* out,
                    const int porder) {
  // porder == 0 : count the non-zero elements of inputs
  if (porder == 0) {
    for (int i = 0; i < pre_n; i++) {
      for (int k = 0; k < post_n; k++) {
        float sum = epsilon;
        const float* in_tmp = input + i * n * post_n + k;
        for (int j = 0; j < n; j++) {
          sum += in_tmp[j * post_n] != 0;
        }
        float* out_tmp = out + i * post_n + k;
        *out_tmp = sum;
      }
    }
  } else {
    for (int i = 0; i < pre_n; i++) {
      for (int k = 0; k < post_n; k++) {
        float sum = epsilon;
        const float* in_tmp = input + i * n * post_n + k;
        for (int j = 0; j < n; j++) {
          sum += std::pow(in_tmp[j * post_n], porder);
        }
        float* out_tmp = out + i * post_n + k;
        *out_tmp = std::pow(sum, 1.f / porder);
      }
    }
  }
}

class PNormComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  int axis_ = 1;
  float epsilon_ = 1e-9;
  float porder_ = 0;
  bool keepdim_ = false;
  DDim dims_{{3, 5, 4, 4}};
  bool bias_after_scale_;

 public:
  PNormComputeTester(const Place& place,
                     const std::string& alias,
                     int axis,
                     float epsilon,
                     float porder,
                     bool keepdim,
                     DDim dims)
      : TestCase(place, alias),
        axis_(axis),
        epsilon_(epsilon),
        porder_(porder),
        keepdim_(keepdim),
        dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    int axis = axis_ < 0 ? axis_ + dims_.size() : axis_;
    int pre_n = dims_.count(0, axis);
    int n = dims_[axis];
    int post_n = dims_.count(axis + 1, dims_.size());

    auto* out = scope->NewTensor(output_);
    CHECK(out);
    std::vector<int64_t> out_shape;
    for (uint8_t i = 0; i < dims_.size(); i++) {
      if (i != axis) out_shape.push_back(dims_[i]);
    }
    out->Resize(DDim(out_shape));
    auto* out_data = out->mutable_data<float>();
    p_norm_compute(x_data, pre_n, n, post_n, epsilon_, out_data, porder_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("p_norm");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("epsilon", epsilon_);
    op_desc->SetAttr("porder", porder_);
    op_desc->SetAttr("keepdim", keepdim_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};
void test_p_norm(Place place, float abs_error) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int axis : {0, 1, 2}) {
    for (float epsilon : {1e-9}) {
      for (float porder : {0, 1, 2}) {
        for (bool keepdim : {false}) {
          std::unique_ptr<arena::TestCase> tester(new PNormComputeTester(
              place, "def", axis, epsilon, porder, keepdim, dims));
          arena::Arena arena(std::move(tester), place, abs_error);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(PNorm, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  test_p_norm(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
