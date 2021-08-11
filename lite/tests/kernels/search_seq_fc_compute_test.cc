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
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {

class SearchSeqFcOPTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string w_ = "w";
  std::string b_ = "b";
  std::string out_ = "out";
  DDim x_dims_;
  DDim w_dims_;
  DDim b_dims_;
  LoD x_lod_;
  bool has_bias_;
  int out_size_;

 public:
  SearchSeqFcOPTest(const Place& place,
                    const std::string& alias,
                    DDim x_dims,
                    DDim w_dims,
                    DDim b_dims,
                    LoD x_lod,
                    bool has_bias,
                    int out_size)
      : TestCase(place, alias),
        x_dims_(x_dims),
        w_dims_(w_dims),
        b_dims_(b_dims),
        x_lod_(x_lod),
        has_bias_(has_bias),
        out_size_(out_size) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto w = scope->FindTensor(w_);
    CHECK(x);
    CHECK(w);
    auto out = scope->NewTensor(out_);
    CHECK(out);

    const auto x_data = x->data<float>();
    const auto w_data = w->data<float>();
    const auto x_dims = x->dims();
    const auto w_dims = w->dims();
    const auto& x_lod = x->lod();
    CHECK_EQ(x_dims.size(), 2) << "The Input(X) should be 2-D tensor.";
    CHECK(!x_lod.empty()) << "The Input(X) must hold lod info.";
    const auto& x_lod_0 = x_lod[0];
    CHECK_GE(x_lod_0.size(), 2) << "The Input(X)'s lod info is corrupted.";
    CHECK_EQ(x_dims[0], static_cast<int64_t>(x_lod_0.back()))
        << "The Input(X)'s lod info mismatches the actual tensor shape.";
    CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor.";
    CHECK_EQ(x_dims[1], w_dims[1]) << "Wrong shape: x_dims[1] != w_dims[1]";
    CHECK_EQ(w_dims[0], out_size_) << "Wrong shape: w_dims[0] != out_size";

    const float* b_data = nullptr;
    if (has_bias_) {
      auto b = scope->FindTensor(b_);
      CHECK(b);
      auto b_dims = b->dims();
      CHECK_EQ(b_dims.size(), 1) << "b should be 1-D tensor.";
      CHECK_EQ(b_dims[0], w_dims[0]) << "Wrong shape: b_dims[0] != w_dims[0]";
      b_data = b->data<float>();
    }

    out->set_lod(x_lod);
    DDim out_dims({x_dims[0], w_dims[0]});
    out->Resize(out_dims);

    int M = x_dims[0];
    int K = x_dims[1];
    int N = w_dims[0];
    auto out_data = out->mutable_data<float>();
    // Prevent 0*nan=nan in basic_gemm
    int64_t out_num = out_dims.production();
    for (int64_t i = 0; i < out_num; i++) {
      out_data[i] = 0;
    }
    basic_gemm<float, float>(false,
                             true,
                             M,
                             N,
                             K,
                             1.f,
                             x_data,
                             K,
                             w_data,
                             K,
                             0,
                             out_data,
                             N,
                             nullptr,
                             false,
                             false);
    if (b_data != nullptr) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          out_data[i * N + j] += b_data[j];
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("search_seq_fc");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("W", {w_});
    if (has_bias_) {
      op_desc->SetInput("b", {b_});
    }
    op_desc->SetAttr<bool>("has_bias", has_bias_);
    op_desc->SetAttr<int>("out_size", out_size_);
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    std::vector<float> w_data(w_dims_.production());
    fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
    fill_data_rand(w_data.data(), -1.f, 1.f, w_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data(), x_lod_);
    SetCommonTensor(w_, w_dims_, w_data.data());
    if (has_bias_) {
      std::vector<float> b_data(b_dims_.production());
      fill_data_rand(b_data.data(), -1.f, 1.f, b_dims_.production());
      SetCommonTensor(b_, b_dims_, b_data.data());
    }
  }
};

void test_search_seq_fc(Place place) {
  for (auto x_lod_0 : {std::vector<uint64_t>({0, 1, 3}),
                       std::vector<uint64_t>({0, 3, 4, 5})}) {
    for (auto feature_size : {2, 9}) {
      for (auto out_size : {3, 5}) {
        for (auto has_bias : {true, false}) {
          DDim x_dims({static_cast<int64_t>(x_lod_0.back()), feature_size});
          DDim w_dims({out_size, feature_size});
          DDim b_dims({has_bias ? out_size : 0});
          LoD x_lod;
          x_lod.push_back(x_lod_0);
          std::unique_ptr<arena::TestCase> tester(new SearchSeqFcOPTest(
              place, "def", x_dims, w_dims, b_dims, x_lod, has_bias, out_size));
          arena::Arena arena(std::move(tester), place, 6e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(SearchSeqFcOP, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  test_search_seq_fc(place);
#endif
}

}  // namespace lite
}  // namespace paddle
