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
#include <cmath>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class MulComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string type_ = "mul";
  std::string x_ = "x";
  std::string y_ = "y";
  std::string out_ = "out";
  DDim x_dims_{{1, 2}};
  DDim y_dims_{{2, 1}};
  int x_num_col_dims_{1};
  int y_num_col_dims_{1};

 public:
  MulComputeTester(const Place& place,
                   const std::string& alias,
                   DDim x_dims,
                   DDim y_dims,
                   int x_num_col_dims,
                   int y_num_col_dims)
      : TestCase(place, alias),
        x_dims_(x_dims),
        y_dims_(y_dims),
        x_num_col_dims_(x_num_col_dims),
        y_num_col_dims_(y_num_col_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* y = scope->FindTensor(y_);
    auto x_mat_dims = x_dims_.Flatten2D(x_num_col_dims_);
    auto y_mat_dims = y_dims_.Flatten2D(y_num_col_dims_);
    CHECK_EQ(x_mat_dims[1], y_mat_dims[0]);

    auto* out = scope->NewTensor(out_);
    CHECK(out);
    std::vector<int64_t> out_shape;
    for (int i = 0; i < x_num_col_dims_; i++) {
      out_shape.push_back(x_dims_[i]);
    }
    for (int i = y_num_col_dims_; i < y_dims_.size(); i++) {
      out_shape.push_back(y_dims_[i]);
    }
    out->Resize(DDim(out_shape));

    auto x_data = x->data<float>();
    auto y_data = y->data<float>();
    auto* out_data = out->mutable_data<float>();

    const int M = x_mat_dims[0];
    const int K = x_mat_dims[1];
    const int N = y_mat_dims[1];
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out_data[m * N + n] = 0;
        for (int k = 0; k < K; ++k) {
          out_data[m * N + n] += x_data[m * K + k] * y_data[k * N + n];
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("x_num_col_dims", x_num_col_dims_);
    op_desc->SetAttr("y_num_col_dims", y_num_col_dims_);
  }

  void PrepareData() override {
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());

    std::vector<float> y(y_dims_.production());
    fill_data_rand(y.data(), -1.f, 1.f, y_dims_.production());
    SetCommonTensor(y_, y_dims_, y.data(), {}, true);
  }
};

void TestMul(const std::vector<int64_t>& x_dims,
             const std::vector<int64_t>& y_dims,
             int x_num_col_dims,
             int y_num_col_dims,
             const Place& place,
             float abs_error) {
  LOG(INFO) << "run test arm";
  std::unique_ptr<arena::TestCase> tester(new MulComputeTester(place,
                                                               "def",
                                                               DDim(x_dims),
                                                               DDim(y_dims),
                                                               x_num_col_dims,
                                                               y_num_col_dims));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Mul, precision) {
  LOG(INFO) << "test mul op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#else
  return;
#endif
  TestMul({4, 5}, {5, 4}, 1, 1, place, abs_error);
  TestMul({4, 5}, {5, 4, 3, 2}, 1, 1, place, abs_error);
  TestMul({4, 20}, {5, 4, 3, 2}, 1, 2, place, abs_error);
  TestMul({4, 60}, {5, 4, 3, 2}, 1, 3, place, abs_error);
  TestMul({2, 3, 4, 5}, {60, 4}, 1, 1, place, abs_error);
  TestMul({2, 3, 4, 5}, {20, 4}, 2, 1, place, abs_error);
  TestMul({2, 3, 4, 5}, {5, 4}, 3, 1, place, abs_error);
  TestMul({2, 3, 4, 5}, {60, 3, 4, 5}, 1, 1, place, abs_error);
  TestMul({2, 3, 4, 5}, {4, 5, 6, 2}, 2, 2, place, abs_error);
  TestMul({2, 3, 4, 5}, {5, 1, 4, 2}, 3, 2, place, abs_error);
}

}  // namespace lite
}  // namespace paddle
