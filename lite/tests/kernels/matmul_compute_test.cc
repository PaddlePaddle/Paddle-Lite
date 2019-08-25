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

namespace paddle {
namespace lite {

void matrix_mul(int m_,
                int k_,
                int n_,
                float alpha,
                const float* x,
                const float* y,
                float* out) {
  for (int m = 0; m < m_; ++m) {
    for (int n = 0; n < n_; ++n) {
      out[m * n_ + n] = 0;
      for (int k = 0; k < k_; ++k) {
        out[m * n_ + n] += x[m * k_ + k] * y[k * n_ + n] * alpha;
      }
    }
  }
}

void transpose(int m, int n, const float* src, float* dst) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      dst[j * m + i] = src[i * n + j];
    }
  }
}

void mul_low_efficiency(DDim x_dims_,
                        DDim y_dims_,
                        bool x_transpose_,
                        bool y_transpose_,
                        float alpha_,
                        const float* x_data,
                        const float* y_data,
                        float* out_data) {
  if (!x_transpose_ && !y_transpose_) {
    CHECK_EQ(x_dims_[1], y_dims_[0])
        << "not supported x_dims(" << x_dims_ << ") and y_dims(" << y_dims_
        << "), x_transpose is " << x_transpose_ << ", y_transpose is "
        << y_transpose_;
    matrix_mul(
        x_dims_[0], y_dims_[0], y_dims_[1], alpha_, x_data, y_data, out_data);
  } else if (!x_transpose_ && y_transpose_) {
    CHECK_EQ(x_dims_[1], y_dims_[1])
        << "not supported x_dims(" << x_dims_ << ") and y_dims(" << y_dims_
        << "), x_transpose is " << x_transpose_ << ", y_transpose is "
        << y_transpose_;
    float* y_data_trans =
        static_cast<float*>(malloc(sizeof(float) * y_dims_[0] * y_dims_[1]));
    transpose(y_dims_[0], y_dims_[1], y_data, y_data_trans);
    matrix_mul(x_dims_[0],
               x_dims_[1],
               y_dims_[0],
               alpha_,
               x_data,
               y_data_trans,
               out_data);
    free(y_data_trans);
  } else if (x_transpose_ && !y_transpose_) {
    CHECK_EQ(x_dims_[0], y_dims_[0])
        << "not supported x_dims(" << x_dims_ << ") and y_dims(" << y_dims_
        << "), x_transpose is " << x_transpose_ << ", y_transpose is "
        << y_transpose_;
    float* x_data_trans =
        static_cast<float*>(malloc(sizeof(float) * x_dims_[0] * x_dims_[1]));
    transpose(x_dims_[0], x_dims_[1], x_data, x_data_trans);
    matrix_mul(x_dims_[1],
               x_dims_[0],
               y_dims_[1],
               alpha_,
               x_data_trans,
               y_data,
               out_data);
    free(x_data_trans);
  } else {
    CHECK_EQ(x_dims_[0], y_dims_[1])
        << "not supported x_dims(" << x_dims_ << ") and y_dims(" << y_dims_
        << "), x_transpose is " << x_transpose_ << ", y_transpose is "
        << y_transpose_;
    float* x_data_trans =
        static_cast<float*>(malloc(sizeof(float) * x_dims_[0] * x_dims_[1]));
    float* y_data_trans =
        static_cast<float*>(malloc(sizeof(float) * y_dims_[0] * y_dims_[1]));
    transpose(x_dims_[0], x_dims_[1], x_data, x_data_trans);
    transpose(y_dims_[0], y_dims_[1], y_data, y_data_trans);
    matrix_mul(x_dims_[1],
               x_dims_[0],
               y_dims_[0],
               alpha_,
               x_data_trans,
               y_data_trans,
               out_data);
    free(x_data_trans);
    free(y_data_trans);
  }
}

class MatMulComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string y_ = "Y";
  std::string out_ = "Out";
  DDim x_dims_;
  DDim y_dims_;
  bool x_transpose_;
  bool y_transpose_;
  float alpha_;

 public:
  MatMulComputeTester(const Place& place,
                      const std::string& alias,
                      bool x_transpose,
                      bool y_transpose,
                      float alpha,
                      const DDim& x_dims,
                      const DDim& y_dims)
      : TestCase(place, alias),
        x_transpose_(x_transpose),
        y_transpose_(y_transpose),
        alpha_(alpha),
        x_dims_(x_dims),
        y_dims_(y_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* y = scope->FindTensor(y_);
    CHECK(x);
    CHECK(y);
    const auto* x_data = x->data<float>();
    const auto* y_data = y->data<float>();
    auto* out = scope->NewTensor(out_);
    CHECK(out);

    std::vector<int64_t> dim_out_vec;
    if (x_dims_.size() > 2 && y_dims_.size() >= 2) {
      // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
      // x: [B, M, K], y: [K, N], out: [B, M, N]
      dim_out_vec.resize(x_dims_.size());
      for (size_t i = 0; i < x_dims_.size() - 2; ++i) {
        dim_out_vec[i] = x_dims_[i];
      }
      if (!x_transpose_ && !y_transpose_) {
        dim_out_vec[x_dims_.size() - 2] = x_dims_[x_dims_.size() - 2];
        dim_out_vec[x_dims_.size() - 1] = y_dims_[y_dims_.size() - 1];
      } else if (!x_transpose_ && y_transpose_) {
        dim_out_vec[x_dims_.size() - 2] = x_dims_[x_dims_.size() - 2];
        dim_out_vec[x_dims_.size() - 1] = y_dims_[y_dims_.size() - 2];
      } else if (x_transpose_ && !y_transpose_) {
        dim_out_vec[x_dims_.size() - 2] = x_dims_[x_dims_.size() - 1];
        dim_out_vec[x_dims_.size() - 1] = y_dims_[y_dims_.size() - 1];
      } else {
        dim_out_vec[x_dims_.size() - 2] = x_dims_[x_dims_.size() - 1];
        dim_out_vec[x_dims_.size() - 1] = y_dims_[y_dims_.size() - 2];
      }

      out->Resize(dim_out_vec);
      auto* out_data = out->mutable_data<float>();
      int x_inner = x_dims_[x_dims_.size() - 2] * x_dims_[x_dims_.size() - 1];

      if (y_dims_.size() > 2) {
        int y_inner = y_dims_[y_dims_.size() - 2] * y_dims_[y_dims_.size() - 1];
        int o_inner =
            dim_out_vec[x_dims_.size() - 2] * dim_out_vec[x_dims_.size() - 1];
        for (size_t i = 0; i < x_dims_.count(0, x_dims_.size() - 2); ++i) {
          mul_low_efficiency(
              DDim({x_dims_[x_dims_.size() - 2], x_dims_[x_dims_.size() - 1]}),
              DDim({y_dims_[y_dims_.size() - 2], y_dims_[y_dims_.size() - 1]}),
              x_transpose_,
              y_transpose_,
              alpha_,
              x_data + i * x_inner,
              y_data + i * y_inner,
              out_data + i * o_inner);
        }
      } else {
        int o_inner =
            dim_out_vec[x_dims_.size() - 2] * dim_out_vec[x_dims_.size() - 1];
        for (size_t i = 0; i < x_dims_.count(0, x_dims_.size() - 2); ++i) {
          mul_low_efficiency(
              DDim({x_dims_[x_dims_.size() - 2], x_dims_[x_dims_.size() - 1]}),
              y_dims_,
              x_transpose_,
              y_transpose_,
              alpha_,
              x_data + i * x_inner,
              y_data,
              out_data + i * o_inner);
        }
      }
    } else if (x_dims_.size() == 2 && y_dims_.size() == 2) {
      // x: [M, K], y: [K, N], out: [M, N]
      dim_out_vec.resize(x_dims_.size());
      if (x_transpose_) {
        dim_out_vec[0] = x_dims_[1];
      } else {
        dim_out_vec[0] = x_dims_[0];
      }
      if (y_transpose_) {
        dim_out_vec[1] = y_dims_[0];
      } else {
        dim_out_vec[1] = y_dims_[1];
      }
      out->Resize(dim_out_vec);
      auto* out_data = out->mutable_data<float>();
      mul_low_efficiency(x_dims_,
                         y_dims_,
                         x_transpose_,
                         y_transpose_,
                         alpha_,
                         x_data,
                         y_data,
                         out_data);
    } else if (x_dims_.size() > 2 && y_dims_.size() == 1) {
      // x: [B, M, K], y: [K], out: [B, M]
      CHECK_EQ(x_dims_[x_dims_.size() - 1], y_dims_[0])
          << "not supported x_dims(" << x_dims_ << ") and y_dims(" << y_dims_
          << ")";
      dim_out_vec.resize(x_dims_.size() - 1);
      for (size_t i = 0; i < dim_out_vec.size(); ++i) {
        dim_out_vec[i] = x_dims_[i];
      }
      out->Resize(dim_out_vec);
      auto* out_data = out->mutable_data<float>();
      for (size_t i = 0; i < x_dims_.count(0, x_dims_.size() - 1); ++i) {
        out_data[i] = 0;
        for (size_t j = 0; j < y_dims_[0]; ++j) {
          out_data[i] += x_data[i * y_dims_[0] + j] * y_data[j] * alpha_;
        }
      }
    } else if (x_dims_.size() == 1 && y_dims_.size() == 1) {
      // x: [K], y: [K], out: [1]
      if (x_dims_[0] == y_dims_[0] && x_transpose_ == false &&
          y_transpose_ == false) {
        dim_out_vec.resize(1);
        dim_out_vec[0] = 1;

        out->Resize(dim_out_vec);
        auto* out_data = out->mutable_data<float>();
        out_data[0] = 0.f;
        for (size_t i = 0; i < x_dims_[0]; ++i) {
          out_data[0] += x_data[i] * y_data[i] * alpha_;
        }
      }
      // x: [M], y: [N], x_transpose: true, y_transpose: true, out: [M, N]
      if (x_transpose_ == true && y_transpose_ == true) {
        dim_out_vec.resize(2);
        dim_out_vec[0] = x_dims_[0];
        dim_out_vec[1] = y_dims_[0];
        out->Resize(dim_out_vec);
        auto* out_data = out->mutable_data<float>();
        mul_low_efficiency(DDim({x_dims_[0], 1}),
                           DDim({1, y_dims_[0]}),
                           false,
                           false,
                           alpha_,
                           x_data,
                           y_data,
                           out_data);
      }
    } else {
      LOG(FATAL) << "not supported x_dims(" << x_dims_ << ") and y_dims("
                 << y_dims_ << ")";
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("matmul");
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

    for (int i = 0; i < x_dims_.production(); ++i) {
      x_data[i] = 1;  // i * 1.1;
    }
    for (int i = 0; i < y_dims_.production(); ++i) {
      y_data[i] = 1;  // i * 0.9;
    }

    SetCommonTensor(x_, x_dims_, x_data.data());
    SetCommonTensor(y_, y_dims_, y_data.data());
  }
};

void test_matmul2x2_no_transform(Place place) {
  for (int m : {1, 2, 4, 8}) {
    for (int k : {1, 3, 5}) {
      for (int n : {1, 2, 4, 6}) {
        for (float alpha : {1., 2.}) {
          bool x_transform = false;
          bool y_transform = false;
          std::unique_ptr<arena::TestCase> tester(
              new MatMulComputeTester(place,
                                      "def",
                                      x_transform,
                                      y_transform,
                                      alpha,
                                      DDim({m, k}),
                                      DDim({k, n})));
          arena::Arena arena(std::move(tester), place, 5e-4);
          arena.TestPrecision();
        }
      }
    }
  }
}

void test_matmul2x2_x_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4}), DDim({2, 5})});
  std::vector<DDim> y_dims({DDim({3, 2}), DDim({2, 1})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", true, false, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

void test_matmul2x2_y_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({5, 2}), DDim({2, 5})});
  std::vector<DDim> y_dims({DDim({3, 2}), DDim({1, 5})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", false, true, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

void test_matmul2x2_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({6, 2}), DDim({5, 3})});
  std::vector<DDim> y_dims({DDim({3, 6}), DDim({1, 5})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", true, true, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 5e-5);
    arena.TestPrecision();
  }
}

void test_matmul1x1_no_transpose(Place place) {
  DDim x_dim({3});
  DDim y_dim({3});
  float alpha = 1.5f;
  std::unique_ptr<arena::TestCase> tester(
      new MatMulComputeTester(place, "def", false, false, alpha, x_dim, y_dim));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

void test_matmul1x1_transpose(Place place) {
  DDim x_dim({3});
  DDim y_dim({5});
  float alpha = 1.5f;
  std::unique_ptr<arena::TestCase> tester(
      new MatMulComputeTester(place, "def", true, true, alpha, x_dim, y_dim));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

void test_matmul_nx1(Place place) {
  DDim x_dim({3, 4, 2, 5});
  DDim y_dim({5});
  float alpha = 1.5f;
  std::unique_ptr<arena::TestCase> tester(
      new MatMulComputeTester(place, "def", false, false, alpha, x_dim, y_dim));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

void test_matmul_nx2_1(Place place) {
  DDim x_dim({1, 2, 2, 3});
  DDim y_dim({3, 1});
  float alpha = 1.f;
  std::unique_ptr<arena::TestCase> tester(
      new MatMulComputeTester(place, "def", false, false, alpha, x_dim, y_dim));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

void test_matmul_nx2_2(Place place) {
  DDim x_dim({1, 2, 2, 3});
  DDim y_dim({3, 3});
  float alpha = 1.5f;
  std::unique_ptr<arena::TestCase> tester(
      new MatMulComputeTester(place, "def", false, false, alpha, x_dim, y_dim));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

void test_matmulnx2_x_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4, 6, 2}), DDim({5, 3, 5, 2})});
  std::vector<DDim> y_dims({DDim({6, 2}), DDim({5, 1})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", true, false, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 2e-4);
    arena.TestPrecision();
  }
}

void test_matmulnx2_y_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4, 6, 2}), DDim({5, 3, 5, 2})});
  std::vector<DDim> y_dims({DDim({6, 2}), DDim({1, 2})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", false, true, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 5e-5);
    arena.TestPrecision();
  }
}

void test_matmulnx2_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4, 4, 3}), DDim({5, 3, 3, 2})});
  std::vector<DDim> y_dims({DDim({2, 4}), DDim({1, 3})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", true, true, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 5e-5);
    arena.TestPrecision();
  }
}

void test_matmul_nxn(Place place) {
  DDim x_dim({3, 4, 2, 5});
  DDim y_dim({3, 4, 5, 2});
  float alpha = 1.5f;
  std::unique_ptr<arena::TestCase> tester(
      new MatMulComputeTester(place, "def", false, false, alpha, x_dim, y_dim));
  arena::Arena arena(std::move(tester), place, 1e-3);
  arena.TestPrecision();
}

void test_matmulnxn_x_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4, 6, 2}), DDim({5, 3, 5, 2})});
  std::vector<DDim> y_dims({DDim({3, 4, 6, 2}), DDim({5, 3, 5, 1})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", true, false, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 1e-3);
    arena.TestPrecision();
  }
}

void test_matmulnxn_y_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4, 6, 2}), DDim({5, 3, 5, 2})});
  std::vector<DDim> y_dims({DDim({3, 4, 6, 2}), DDim({5, 3, 1, 2})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", false, true, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 1e-3);
    arena.TestPrecision();
  }
}

void test_matmulnxn_transpose(Place place) {
  std::vector<DDim> x_dims({DDim({3, 4, 4, 3}), DDim({5, 3, 3, 2})});
  std::vector<DDim> y_dims({DDim({3, 4, 2, 4}), DDim({5, 3, 1, 3})});
  std::vector<float> alphas({1.f, 2.f});
  for (int i = 0; i < x_dims.size(); ++i) {
    std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(
        place, "def", true, true, alphas[i], x_dims[i], y_dims[i]));
    arena::Arena arena(std::move(tester), place, 1e-3);
    arena.TestPrecision();
  }
}

TEST(Matmul2x2, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul2x2_no_transform(place);
#endif
}

TEST(Matmul2x2_x_transpose, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul2x2_x_transpose(place);
#endif
}
TEST(Matmul2x2_y_transpose, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul2x2_y_transpose(place);
#endif
}

TEST(Matmul2x2_transpose, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul2x2_transpose(place);
#endif
}

TEST(Matmul1x1, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul1x1_transpose(place);
  test_matmul1x1_no_transpose(place);
#endif
}

TEST(Matmulnx1, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul_nx1(place);
#endif
}

TEST(Matmulnx2, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul_nx2_1(place);
  test_matmul_nx2_2(place);
  test_matmulnx2_x_transpose(place);
  test_matmulnx2_y_transpose(place);
  test_matmulnx2_transpose(place);
#endif
}

TEST(Matmulnxn, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_matmul_nxn(place);
  test_matmulnxn_x_transpose(place);
  test_matmulnxn_y_transpose(place);
  test_matmulnxn_transpose(place);
#endif
}

}  // namespace lite
}  // namespace paddle
