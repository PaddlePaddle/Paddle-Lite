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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class SoftmaxComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "softmax";
  std::string input_ = "x";
  std::string output_ = "out";
  DDim dims_{{1, 2, 3, 4}};
  int axis_ = 1;

 public:
  SoftmaxComputeTest(const Place& place,
                     const std::string& alias,
                     DDim dims,
                     int axis)
      : TestCase(place, alias), dims_(dims), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();
    auto x_rank = dims_.size();
    if (axis_ < 0) {
      axis_ += x_rank;
    }
    int axis_size = dims_[axis_];
    int outer_num = dims_.Slice(0, axis_).production();
    int inner_num = dims_.Slice(axis_ + 1, x_rank).production();
    int compute_size = outer_num * inner_num;
    for (int i = 0; i < compute_size; i++) {
      int idx_inner = i % inner_num;
      int idx_outer = (i / inner_num) * axis_size;
      int start = idx_outer * inner_num + idx_inner;
      int offset;

      offset = start;
      float max_data = std::numeric_limits<float>::lowest();
      for (int j = 0; j < axis_size; j++) {
        max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
        offset += inner_num;
      }

      offset = start;
      float sum_data = 0.f;
      for (int j = 0; j < axis_size; j++) {
        out_data[offset] = exp(x_data[offset] - max_data);
        sum_data += out_data[offset];
        offset += inner_num;
      }

      offset = start;
      for (int j = 0; j < axis_size; j++) {
        out_data[offset] /= sum_data;
        offset += inner_num;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

TEST(Softmax, precision) {
  LOG(INFO) << "test softmax op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#else
  return;
#endif

  std::vector<std::vector<int64_t>> dims{{1, 2, 3, 4}, {2, 3, 4}, {3, 4}};
  for (auto dim_in : dims) {
    for (auto axis : {-1, 0, 1, 2, 3}) {
      if (axis >= dim_in.size()) continue;
      std::unique_ptr<arena::TestCase> tester(
          new SoftmaxComputeTest(place, "def", DDim(dim_in), axis));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

}  // namespace lite
}  // namespace paddle
