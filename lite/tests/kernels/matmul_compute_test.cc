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
                      const DDim& x_dims,
                      const DDim& y_dims,
                      bool x_transpose = false,
                      bool y_transpose = false,
                      float alpha = 1.f)
      : TestCase(place, alias),
        x_dims_(x_dims),
        y_dims_(y_dims),
        x_transpose_(x_transpose),
        y_transpose_(y_transpose),
        alpha_(alpha) {}

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
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());

    std::vector<float> y(y_dims_.production());
    fill_data_rand(y.data(), -1.f, 1.f, y_dims_.production());
    SetCommonTensor(y_, y_dims_, y.data(), {}, true);
  }
};

void test_matmul_helper(Place place,
                        float abs_error,
                        std::vector<int64_t> x_dims,
                        std::vector<int64_t> y_dims,
                        bool x_transpose,
                        bool y_transpose,
                        float alpha) {
  std::unique_ptr<arena::TestCase> tester(new MatMulComputeTester(place,
                                                                  "def",
                                                                  DDim(x_dims),
                                                                  DDim(y_dims),
                                                                  x_transpose,
                                                                  y_transpose,
                                                                  alpha));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void test_matmul2x2(Place place, float abs_error) {
  for (int64_t m : {1, 2, 8}) {
    for (int64_t k : {1, 3, 5}) {
      for (int64_t n : {1, 4, 6}) {
        for (float alpha : {1., 2.}) {
          test_matmul_helper(
              place, abs_error, {m, k}, {k, n}, false, false, alpha);
        }
      }
    }
  }
}

void test_matmul2x2_xtranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(place, abs_error, {3, 4}, {3, 2}, true, false, alpha);
    test_matmul_helper(place, abs_error, {2, 5}, {2, 1}, true, false, alpha);
  }
}

void test_matmul2x2_ytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(place, abs_error, {5, 2}, {3, 2}, false, true, alpha);
    test_matmul_helper(place, abs_error, {2, 4}, {3, 4}, false, true, alpha);
  }
}

void test_matmul2x2_xytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(place, abs_error, {6, 2}, {3, 6}, true, true, alpha);
    test_matmul_helper(place, abs_error, {5, 3}, {1, 5}, true, true, alpha);
  }
}

void test_matmul1x1(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(place, abs_error, {3}, {3}, false, false, alpha);
  }
}

void test_matmul1x1_xytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(place, abs_error, {3}, {5}, true, true, alpha);
  }
}

void test_matmulnx1(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 2, 5}, {5}, false, false, alpha);
  }
}

void test_matmulnx2(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {1, 2, 2, 3}, {3, 1}, false, false, alpha);
    test_matmul_helper(
        place, abs_error, {1, 2, 2, 3}, {3, 4}, false, false, alpha);
  }
}

void test_matmulnx2_xtranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 6, 2}, {6, 2}, true, false, alpha);
    test_matmul_helper(
        place, abs_error, {5, 3, 5, 2}, {5, 1}, true, false, alpha);
  }
}

void test_matmulnx2_ytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 6, 2}, {5, 2}, false, true, alpha);
    test_matmul_helper(
        place, abs_error, {5, 3, 5, 2}, {1, 2}, false, true, alpha);
  }
}

void test_matmulnx2_xytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 4, 3}, {2, 4}, true, true, alpha);
    test_matmul_helper(
        place, abs_error, {5, 3, 3, 2}, {1, 3}, true, true, alpha);
  }
}

void test_matmulnxn(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 6, 2}, {3, 4, 2, 5}, false, false, alpha);
    test_matmul_helper(
        place, abs_error, {5, 3, 4}, {5, 4, 6}, false, false, alpha);
  }
}

void test_matmulnxn_xtranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 2, 6}, {3, 4, 2, 5}, true, false, alpha);
    test_matmul_helper(
        place, abs_error, {5, 4, 2}, {5, 4, 6}, true, false, alpha);
  }
}

void test_matmulnxn_ytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
#ifdef LITE_WITH_OPENCL
    test_matmul_helper(place, abs_error, {3, 2}, {5, 2}, false, true, alpha);
    test_matmul_helper(place, abs_error, {5, 6}, {8, 6}, false, true, alpha);
#else
    test_matmul_helper(
        place, abs_error, {3, 4, 6, 2}, {3, 4, 5, 2}, false, true, alpha);
    test_matmul_helper(
        place, abs_error, {5, 3, 4}, {5, 6, 4}, false, true, alpha);
#endif
  }
}

void test_matmulnxn_xytranspose(Place place, float abs_error) {
  for (float alpha : {1.f, 2.f}) {
    test_matmul_helper(
        place, abs_error, {3, 4, 2, 6}, {3, 4, 5, 2}, true, true, alpha);
    test_matmul_helper(
        place, abs_error, {5, 4, 3}, {5, 6, 4}, true, true, alpha);
  }
}

TEST(Matmul2x2, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmul2x2(place, abs_error);
}

TEST(Matmul2x2_x_transpose, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmul2x2_xtranspose(place, abs_error);
}

TEST(Matmul2x2_y_transpose, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmul2x2_ytranspose(place, abs_error);
}

TEST(Matmul2x2_transpose, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmul2x2_xytranspose(place, abs_error);
}

TEST(Matmul1x1, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmul1x1(place, abs_error);
  test_matmul1x1_xytranspose(place, abs_error);
}

TEST(Matmulnx1, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmulnx1(place, abs_error);
}

TEST(Matmulnx2, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmulnx2(place, abs_error);
  test_matmulnx2_xtranspose(place, abs_error);
  test_matmulnx2_ytranspose(place, abs_error);
  test_matmulnx2_xytranspose(place, abs_error);
}

#ifdef LITE_WITH_OPENCL
TEST(Matmul, opencl) {
  Place place = TARGET(kOpenCL);
  float abs_error = 2e-4;
  test_matmul2x2(place, abs_error);
  test_matmul2x2_ytranspose(place, abs_error);
}
#endif

TEST(Matmulnxn, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-2;
  test_matmulnxn(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  test_matmulnxn(place, abs_error);
  test_matmulnxn_ytranspose(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  test_matmulnxn(place, abs_error);
  test_matmulnxn_xtranspose(place, abs_error);
  test_matmulnxn_ytranspose(place, abs_error);
  test_matmulnxn_xytranspose(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
