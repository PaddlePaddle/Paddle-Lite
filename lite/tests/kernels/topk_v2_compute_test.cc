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

template <typename T1, typename T2>
bool comp_func(std::pair<T1, T2> a, std::pair<T1, T2> b) {
  return (a.first > b.first);
}

template <typename T1, typename T2>
class TopkV2ComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  std::string indices_ = "indices";
  DDim x_dims_{{3, 5, 4, 4}};
  int axis_ = -1;
  int k_ = 1;

 public:
  TopkV2ComputeTester(const Place& place,
                      const std::string& alias,
                      DDim x_dims,
                      int axis,
                      int k = 1)
      : TestCase(place, alias), x_dims_(x_dims), axis_(axis), k_(k) {}

  void RunBaseline(Scope* scope) override {
    auto* out_val = scope->NewTensor(out_);
    auto* out_ind = scope->NewTensor(indices_);
    DDim out_dims = x_dims_;
    if (axis_ < 0) {
      axis_ += x_dims_.size();
    }
    out_dims[axis_] = k_;
    out_val->Resize(out_dims);
    out_ind->Resize(out_dims);
    auto* out_val_data = out_val->template mutable_data<T1>();
    auto* out_ind_data = out_ind->template mutable_data<T2>();

    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->template data<T1>();
    int inner_size = x_dims_.count(axis_ + 1, x_dims_.size());
    int axis_size = x_dims_[axis_];
    int outer_size = x_dims_.count(0, axis_);
    int out_sum_size = k * inner_size;
    for (int n = 0; n < outer_size; n++) {
      const float* in_data = x_data + n * sum_size;
      float* out_val_data1 = out_val_data + n * out_sum_size;
      int64_t* out_ind_data1 = out_ind_data + n * out_sum_size;
      for (int i = 0; i < inner_size; i++) {
        std::vector<std::pair<float, int>> vec;
        for (int j = 0; j < axis_size; j++) {
          vec.push_back(std::make_pair(in_data[j * outer_size + i], j));
        }
        std::partial_sort(
            vec.begin(), vec.begin() + k_, vec.end(), comp_func<T1, T2>);
        for (int j = 0; j < k_; j++) {
          out_val_data1[j * outer_size + i] = vec[j].first;
          out_ind_data1[j * outer_size + i] = vec[j].second;
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("top_k_v2");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetOutput("Indices", {indices_});
    op_desc->SetAttr("k", k_);
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<T1> dx(x_dims_.production());
    fill_data_rand<T1>(dx.data(), -1, 1, x_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());
  }
};

template <typename T1, typename T2>
void test_topk_v2(Place place, float abs_error) {
  for (auto x_shape :
       std::vector<std::vector<int64_t>>{{2, 3, 4, 5}, {3, 4, 5}, {4, 5}}) {
    for (int axis : {-1, -2}) {
      for (int k : {2, 5}) {
        std::unique_ptr<arena::TestCase> tester(new TopkV2ComputeTester<T1, T2>(
            place, "def", DDim(x_shape), axis, k));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

TEST(Topk, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
  test_topk_v2<float, int64_t>(place, abs_error);
#else
  return;
#endif
}

}  // namespace lite
}  // namespace paddle
