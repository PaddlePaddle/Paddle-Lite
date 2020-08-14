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
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {

class SearchAlignedMatMulComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string y_ = "Y";
  bool x_transpose_;
  bool y_transpose_;
  float alpha_;
  std::string out_ = "Out";
  DDim x_dims_;
  DDim y_dims_;
  LoD x_lod_;
  LoD y_lod_;

 public:
  SearchAlignedMatMulComputeTester(const Place& place,
                                   const std::string& alias,
                                   bool x_transpose,
                                   bool y_transpose,
                                   float alpha,
                                   const DDim& x_dims,
                                   const DDim& y_dims,
                                   const LoD& x_lod,
                                   const LoD& y_lod)
      : TestCase(place, alias),
        x_transpose_(x_transpose),
        y_transpose_(y_transpose),
        alpha_(alpha),
        x_dims_(x_dims),
        y_dims_(y_dims),
        x_lod_(x_lod),
        y_lod_(y_lod) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto y = scope->FindTensor(y_);
    CHECK(x);
    CHECK(y);
    const auto x_data = x->data<float>();
    const auto y_data = y->data<float>();
    auto out = scope->NewTensor(out_);
    CHECK(out);

    const auto x_dims = x->dims();
    const auto y_dims = y->dims();
    const auto& x_lod = x->lod();
    const auto& y_lod = y->lod();
    const auto& x_lod_0 = x_lod[0];
    const auto& y_lod_0 = y_lod[0];

    int seq_num = x_lod_0.size() - 1;
    int x_inner_size = x_dims[1];
    int y_inner_size = y_dims[1];
    int x_batch_size = x_lod_0[1];
    int y_batch_size = y_lod_0[1];
    int M = x_transpose_ ? x_inner_size : x_batch_size;
    int N = y_transpose_ ? y_batch_size : y_inner_size;
    int X_K = x_transpose_ ? x_batch_size : x_inner_size;
    int Y_K = y_transpose_ ? y_inner_size : y_batch_size;
    CHECK_EQ(X_K, Y_K) << "K of Input(X) and Input(Y) is not equal";
    int K = X_K;
    int x_stride = x_batch_size * x_inner_size;
    int y_stride = y_batch_size * y_inner_size;
    int out_stride = M * N;
    int lda = x_transpose_ ? M : K;
    int ldb = y_transpose_ ? K : N;
    int ldc = N;

    LoD out_lod;
    std::vector<uint64_t> out_lod_0(seq_num + 1);
    out_lod_0[0] = 0;
    for (int i = 0; i < seq_num; i++) {
      out_lod_0[i + 1] = out_lod_0[i] + M;
    }
    out_lod.push_back(out_lod_0);
    DDim out_dims(
        {static_cast<int64_t>(out_lod_0.back()), static_cast<int64_t>(N)});
    out->set_lod(out_lod);
    out->Resize(out_dims);

    auto out_data = out->mutable_data<float>();
    // Prevent 0*nan=nan in basic_gemm
    int64_t out_num = out_dims.production();
    for (int64_t i = 0; i < out_num; i++) {
      out_data[i] = 0;
    }
    for (int i = 0; i < seq_num; i++) {
      basic_gemm<float, float>(x_transpose_,
                               y_transpose_,
                               M,
                               N,
                               K,
                               alpha_,
                               x_data + i * x_stride,
                               lda,
                               y_data + i * y_stride,
                               ldb,
                               0,
                               out_data + i * out_stride,
                               ldc,
                               nullptr,
                               false,
                               false);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("search_aligned_mat_mul");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("transpose_X", x_transpose_);
    op_desc->SetAttr("transpose_Y", y_transpose_);
    op_desc->SetAttr("alpha", alpha_);
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    std::vector<float> y_data(y_dims_.production());
    fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
    fill_data_rand(y_data.data(), -1.f, 1.f, y_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data(), x_lod_);
    SetCommonTensor(y_, y_dims_, y_data.data(), y_lod_);
  }
};

void test_search_aligned_mat_mul(Place place) {
  for (int seq_num : {1, 2}) {
    for (int x_batch_size : {1, 3}) {
      for (int x_inner_size : {1, 5}) {
        for (int out_inner_size : {1, 4}) {
          for (bool x_transpose : {true, false}) {
            for (bool y_transpose : {true, false}) {
              for (float alpha : {1., 2.}) {
                // infer x_dims and y_dims
                int y_batch_size;
                int y_inner_size;
                if (x_transpose) {
                  if (y_transpose) {
                    y_batch_size = out_inner_size;
                    y_inner_size = x_batch_size;
                  } else {
                    y_batch_size = x_batch_size;
                    y_inner_size = out_inner_size;
                  }
                } else {
                  if (y_transpose) {
                    y_batch_size = out_inner_size;
                    y_inner_size = x_inner_size;
                  } else {
                    y_batch_size = x_inner_size;
                    y_inner_size = out_inner_size;
                  }
                }
                std::vector<uint64_t> x_lod_0(seq_num + 1);
                std::vector<uint64_t> y_lod_0(seq_num + 1);
                x_lod_0[0] = 0;
                y_lod_0[0] = 0;
                for (int i = 0; i < seq_num; i++) {
                  x_lod_0[i + 1] = x_lod_0[i] + x_batch_size;
                  y_lod_0[i + 1] = y_lod_0[i] + y_batch_size;
                }
                LoD x_lod;
                LoD y_lod;
                x_lod.push_back(x_lod_0);
                y_lod.push_back(y_lod_0);
                DDim x_dims({static_cast<int64_t>(x_lod_0.back()),
                             static_cast<int64_t>(x_inner_size)});
                DDim y_dims({static_cast<int64_t>(y_lod_0.back()),
                             static_cast<int64_t>(y_inner_size)});

                std::unique_ptr<arena::TestCase> tester(
                    new SearchAlignedMatMulComputeTester(place,
                                                         "def",
                                                         x_transpose,
                                                         y_transpose,
                                                         alpha,
                                                         x_dims,
                                                         y_dims,
                                                         x_lod,
                                                         y_lod));
                arena::Arena arena(std::move(tester), place, 5e-4);
                arena.TestPrecision();
              }
            }
          }
        }
      }
    }
  }
}

TEST(SearchAlignedMatMul, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  test_search_aligned_mat_mul(place);
#endif
}

}  // namespace lite
}  // namespace paddle
