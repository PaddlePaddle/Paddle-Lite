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

template <typename T1, typename T2>
bool comp_func(std::pair<T1, T2> a, std::pair<T1, T2> b) {
  return (a.first > b.first);
}

template <typename T1, typename T2>
class TopkComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  std::string indices_ = "indices";
  DDim x_dims_{{3, 5, 4, 4}};
  int k_ = 1;

 public:
  TopkComputeTester(const Place& place,
                    const std::string& alias,
                    DDim x_dims,
                    int k = 1)
      : TestCase(place, alias), x_dims_(x_dims), k_(k) {}

  void RunBaseline(Scope* scope) override {
    auto* out_val = scope->NewTensor(out_);
    auto* out_ind = scope->NewTensor(indices_);
    DDim out_dims = x_dims_;
    out_dims[out_dims.size() - 1] = k_;
    out_val->Resize(out_dims);
    out_ind->Resize(out_dims);
    auto* out_val_data = out_val->template mutable_data<T1>();
    auto* out_ind_data = out_ind->template mutable_data<T2>();

    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->template data<T1>();
    int m = out_dims.production() / k_;
    int n = x_dims_[x_dims_.size() - 1];

    for (int i = 0; i < m; i++) {
      const T1* in_tmp = x_data + i * n;
      T1* out_val_tmp = out_val_data + i * k_;
      T2* out_ind_tmp = out_ind_data + i * k_;
      std::vector<std::pair<T1, T2>> vec;
      for (int j = 0; j < n; j++) {
        vec.push_back(std::make_pair(in_tmp[j], static_cast<T2>(j)));
      }
      std::partial_sort(
          vec.begin(), vec.begin() + k_, vec.end(), comp_func<T1, T2>);
      for (int q = 0; q < k_; q++) {
        out_val_tmp[q] = vec[q].first;
        out_ind_tmp[q] = vec[q].second;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("top_k");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetOutput("Indices", {indices_});
    op_desc->SetAttr("k", k_);
  }

  void PrepareData() override {
    std::vector<T1> dx(x_dims_.production());
    fill_data_rand<T1>(dx.data(), -1, 1, x_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());
  }
};

template <typename T1, typename T2>
void test_topk(Place place, float abs_error) {
  for (auto x_shape : std::vector<std::vector<int64_t>>{
           {2, 3, 4, 5}, {3, 4, 5}, {4, 5}, {5}}) {
    for (int k : {2, 5}) {
      std::unique_ptr<arena::TestCase> tester(
          new TopkComputeTester<T1, T2>(place, "def", DDim(x_shape), k));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(Topk, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-3;  // Using fp16 in NPU
  // TODO(zhupengyang): enable later
  return;
  test_topk<float, int>(place, abs_error);
#elif defined(LITE_WITH_X86) || defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  test_topk<float, int64_t>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
