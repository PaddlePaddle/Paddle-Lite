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

template <typename T>
void CosSim(const T* x_,
            const T* y_,
            T* x_norm_,
            T* y_norm_,
            T* z_,
            size_t cols_,
            size_t row_id,
            bool same_row) {
  auto* x = x_ + cols_ * row_id;
  T xx = 0, xy = 0, yy = 0;
  T eps = 1e-8;
  if (same_row) {
    auto* y = y_ + cols_ * row_id;
    T tep_x, tep_y;
    for (size_t i = 0; i < cols_; ++i) {
      tep_x = x[i];
      tep_y = y[i];
      xx += tep_x * tep_x;

      yy += tep_y * tep_y;
      xy += tep_x * tep_y;
    }
    xx = xx > eps ? xx : eps;
    yy = yy > eps ? yy : eps;
    xx = sqrt(xx);
    yy = sqrt(yy);
    y_norm_[row_id] = yy;
    x_norm_[row_id] = xx;
    z_[row_id] = xy / (xx * yy);
  } else {
    T tep_x, tep_y;
    for (size_t i = 0; i < cols_; ++i) {
      tep_x = x[i];
      tep_y = y_[i];
      xx += tep_x * tep_x;
      yy += tep_y * tep_y;
      xy += tep_x * tep_y;
    }
    xx = xx > eps ? xx : eps;
    yy = yy > eps ? yy : eps;
    xx = sqrt(xx);
    yy = sqrt(yy);
    if (row_id == 0) y_norm_[0] = yy;
    x_norm_[row_id] = xx;
    z_[row_id] = xy / (xx * yy);
  }
}

template <typename T>
class CosSimComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string y_ = "y";
  std::string out_ = "out";
  std::string x_norm_ = "x_norm";
  std::string y_norm_ = "y_norm";
  DDim x_dims_;
  DDim y_dims_;

 public:
  CosSimComputeTester(const Place& place,
                      const std::string& alias,
                      const DDim& x_dims,
                      const DDim& y_dims)
      : TestCase(place, alias), x_dims_(x_dims), y_dims_(y_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* x_norm = scope->NewTensor(x_norm_);
    auto* y_norm = scope->NewTensor(y_norm_);
    out->Resize({x_dims_[0], 1});
    x_norm->Resize({x_dims_[0], 1});
    y_norm->Resize({y_dims_[0], 1});
    auto* x = scope->FindTensor(x_);
    auto* y = scope->FindTensor(y_);

    size_t rows_x = x_dims_[0];
    size_t rows_y = y_dims_[0];
    size_t cols = static_cast<size_t>(x_dims_.production()) / rows_x;
    bool same_row = rows_x == rows_y;
    for (size_t i = 0; i < rows_x; i++) {
      CosSim(x->template data<T>(),
             y->template data<T>(),
             x_norm->template mutable_data<T>(),
             y_norm->template mutable_data<T>(),
             out->template mutable_data<T>(),
             cols,
             i,
             same_row);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("cos_sim");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetOutput("XNorm", {x_norm_});
    op_desc->SetOutput("YNorm", {y_norm_});
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand(x_data.data(),
                   static_cast<T>(-1),
                   static_cast<T>(1),
                   x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());

    std::vector<T> y_data(y_dims_.production());
    fill_data_rand(y_data.data(),
                   static_cast<T>(-1),
                   static_cast<T>(1),
                   y_dims_.production());
    SetCommonTensor(y_, y_dims_, y_data.data());
  }
};

template <typename T>
void TestCosSim(Place place, float abs_error) {
  for (auto x_shape :
       std::vector<std::vector<int64_t>>{{3, 4, 5, 6}, {3, 4, 5}, {4, 5}}) {
    for (bool same_row : {true, false}) {
      std::vector<int64_t> y_shape = x_shape;
      if (!same_row) y_shape[0] = 1;
      std::unique_ptr<arena::TestCase> tester(new CosSimComputeTester<T>(
          place, "def", DDim(x_shape), DDim(y_shape)));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(cos_sim, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestCosSim<float>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
