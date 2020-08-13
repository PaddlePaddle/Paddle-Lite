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
#ifdef LITE_WITH_X86
#include "lite/backends/x86/parallel.h"
#endif

namespace paddle {
namespace lite {

void AddBias(float* out, const float* bias, int num, int channel) {
  int remain = channel;
  for (int j = 0; j < num; ++j) {
    const float* ptr_bias = bias;
    float* ptr_out = out + j * channel;
    for (int i = 0; i < remain; ++i) {
      *(ptr_out++) += *(ptr_bias++);
    }
  }
}

void Relu(float* out, int num, int channel) {
  for (int i = 0; i < num * channel; ++i) {
    if (out[i] < 0) {
      out[i] = 0;
    }
  }
}

DDim ComputeOutDim(const DDim& dim_in, const DDim& wdim, int in_num_col_dim) {
  std::vector<int64_t> out_dim;
  out_dim.resize(in_num_col_dim + 1);
  for (int i = 0; i < in_num_col_dim; ++i) {
    out_dim[i] = dim_in[i];
  }
  out_dim[in_num_col_dim] = wdim[1];
  return DDim(out_dim);
}

class FcOPTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string weight_ = "w";
  std::string weight_padding_ = "w_padding";
  std::string bias_ = "b";
  std::string out_ = "out";
  DDim dims_{{1, 128}};
  DDim wdims_{{128, 4}};
  DDim wdims_padding_;
  DDim bdims_{{4}};
  int in_num_col_dims_{1};
  bool with_relu_{false};
  bool padding_weights_{false};

 public:
  FcOPTest(const Place& place,
           const std::string& alias,
           DDim dim_in,
           DDim dim_w,
           DDim dim_b,
           int in_num_col_dims,
           bool with_relu,
           bool padding)
      : TestCase(place, alias),
        dims_(std::move(dim_in)),
        wdims_(std::move(dim_w)),
        bdims_(dim_b),
        in_num_col_dims_(in_num_col_dims),
        with_relu_(with_relu) {
#ifdef LITE_WITH_X86
    if (padding && wdims_[0] % 128 == 0 && wdims_[1] % 128 == 0) {
      padding_weights_ = true;
      wdims_padding_ = DDim({wdims_[0] + 4, wdims_[1] + 4});
    }
#endif
  }

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto w = scope->FindTensor(weight_);
    auto b = scope->FindTensor(bias_);
    bool flag_bias = b;
    auto out = scope->NewTensor(out_);
    CHECK(out);
    DDim out_dim = ComputeOutDim(x->dims(), w->dims(), in_num_col_dims_);
    out->Resize(out_dim);

    auto x_data = x->data<float>();
    auto w_data = w->data<float>();
    const float* b_data = nullptr;
    if (flag_bias) {
      b_data = b->data<float>();
    }
    auto out_data = out->mutable_data<float>();

    // must init out_data to be 0 firstly
    for (int i = 0; i < out_dim.production(); i++) {
      out_data[i] = 0;
    }

    int m = x->dims().count(0, in_num_col_dims_);
    CHECK_EQ(wdims_[0], x->dims().count(in_num_col_dims_, x->dims().size()));
    int k = wdims_[0];
    int n = wdims_[1];

    LOG(INFO) << "M=" << m << ", N=" << n << ", K=" << k
              << ", bias=" << flag_bias << ", with_relu=" << with_relu_
              << ", padding_weights=" << padding_weights_;

    if (m == 1) {
      basic_gemv(n,
                 k,
                 w_data,
                 x_data,
                 b_data,
                 out_data,
                 1.f,
                 0.f,
                 true,
                 static_cast<int>(flag_bias),
                 false);
    } else {
      basic_gemm(false,
                 false,
                 m,
                 n,
                 k,
                 1.f,
                 x_data,
                 k,
                 w_data,
                 n,
                 0.f,
                 out_data,
                 n,
                 b_data,
                 false,
                 false);
      if (flag_bias) {
        AddBias(out_data, b_data, m, n);
      }
    }
#ifdef LITE_WITH_X86
    if (flag_bias && with_relu_) {
      Relu(out_data, m, n);
    }
#endif
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fc");
    op_desc->SetInput("Input", {input_});
    if (padding_weights_) {
      op_desc->SetInput("W", {weight_padding_});
    } else {
      op_desc->SetInput("W", {weight_});
    }
    if (bdims_.production() > 0) {
      op_desc->SetInput("Bias", {bias_});
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr<int>("in_num_col_dims", in_num_col_dims_);
#ifdef LITE_WITH_X86
    std::string activation_type = with_relu_ ? "relu" : "";
    op_desc->SetAttr<std::string>("activation_type", activation_type);
    op_desc->SetAttr<bool>("padding_weights", padding_weights_);
#endif
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());

    std::vector<float> win(wdims_.production());
    fill_data_rand(win.data(), -1.f, 1.f, wdims_.production());

    bool flag_bias = bdims_.production() > 0;
    std::vector<float> bin(bdims_.production());
    fill_data_rand(bin.data(), -1.f, 1.f, bdims_.production());

    SetCommonTensor(input_, dims_, din.data());
    SetCommonTensor(weight_, wdims_, win.data(), {}, true);
    if (padding_weights_) {
      std::vector<float> win_padding(wdims_padding_.production());
      for (int64_t i = 0; i < wdims_[0]; ++i) {
        memcpy(&(win_padding[i * wdims_padding_[1]]),
               &(win[i * wdims_[1]]),
               wdims_[1] * sizeof(float));
      }
      SetCommonTensor(weight_padding_, wdims_padding_, win_padding.data());
    }
    if (flag_bias) {
      SetCommonTensor(bias_, bdims_, bin.data(), {}, true);
    }
  }
};

void TestFC2D(Place place,
              float abs_error,
              bool with_relu = false,
              bool padding = false) {
  for (auto& m : {1, 3, 16}) {
    for (auto& n : {1, 4, 16, 128, 256, 1024}) {
      for (auto& k : {1, 16, 128, 1024}) {
        for (auto& bflag : {false, true}) {
          if (!bflag && with_relu) {
            continue;
          }
          DDim dim_in{{m, k}};
          DDim wdim{{k, n}};
          DDim bdim{{bflag ? n : 0}};
          std::unique_ptr<arena::TestCase> tester(new FcOPTest(
              place, "def", dim_in, wdim, bdim, 1, with_relu, padding));
#ifdef LITE_WITH_ARM
          if (place == TARGET(kARM)) {
            auto& ctx = tester->context()->As<ARMContext>();
            ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 1);
          }
#endif
          arena::Arena arena(std::move(tester), place, abs_error);
          if (!arena.TestPrecision()) {
            LOG(ERROR) << "run m: " << m << ", n: " << n << ", k: " << k
                       << ", bias: " << (bflag ? "true" : "false") << " failed";
            return;
          }
        }
      }
    }
  }
}

void TestFCHelper(Place place,
                  float abs_error,
                  std::vector<int64_t> xdims,
                  std::vector<int64_t> wdims,
                  std::vector<int64_t> bdims,
                  int in_num_col_dims) {
  std::unique_ptr<arena::TestCase> tester(new FcOPTest(place,
                                                       "def",
                                                       DDim(xdims),
                                                       DDim(wdims),
                                                       DDim(bdims),
                                                       in_num_col_dims,
                                                       false,
                                                       false));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestFCnD(Place place, float abs_error) {
  TestFCHelper(place, abs_error, {2, 3, 4}, {4, 5}, {5}, 2);
  TestFCHelper(place, abs_error, {2, 3, 4}, {12, 5}, {5}, 1);
  TestFCHelper(place, abs_error, {2, 3, 4, 5}, {5, 6}, {6}, 3);
  TestFCHelper(place, abs_error, {2, 3, 4, 5}, {20, 6}, {6}, 2);
  TestFCHelper(place, abs_error, {2, 3, 4, 5}, {60, 6}, {6}, 1);
}

TEST(FcOP, precision) {
  Place place;
  float abs_error = 1e-4;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 2e-1;  // Using fp16 in NPU
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
  abs_error = 1e-4;
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  TestFC2D(place, abs_error);
  TestFCnD(place, abs_error);
}

#ifdef LITE_WITH_X86
TEST(FcOP, padding_and_parallel) {
  Place place(TARGET(kX86));
  float abs_error = 1e-4;
  x86::SetNumThreads(4);
  TestFC2D(place, abs_error, true, true);
}
#endif

}  // namespace lite
}  // namespace paddle
