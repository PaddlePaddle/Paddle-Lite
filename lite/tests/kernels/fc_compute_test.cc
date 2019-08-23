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
#include "lite/tests/kernels/fill_data.h"
#include "lite/tests/kernels/test_funcs.h"

namespace paddle {
namespace lite {

void fill_bias_fc(float* out, const float* bias, int num, int channel) {
  int remain = channel;
  for (int j = 0; j < num; ++j) {
    const float* ptr_bias = bias;
    float* ptr_out = out + j * channel;
    for (int i = 0; i < remain; ++i) {
      *(ptr_out++) += *(ptr_bias++);
    }
  }
}

DDim compute_out_dim(const DDim& dim_in, const DDim& wdim, int in_num_col_dim) {
  std::vector<int64_t> out_dim;
  out_dim.resize(in_num_col_dim + 1);
  auto in_mat_dims = dim_in.Flatten2D(in_num_col_dim);
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
  std::string bias_ = "b";
  std::string out_ = "out";
  int in_num_col_dims_{1};
  DDim dims_{{1, 128}};
  DDim wdims_{{128, 4}};
  DDim bdims_{{4}};

 public:
  FcOPTest(const Place& place,
           const std::string& alias,
           DDim dim_in,
           DDim dim_w,
           DDim dim_b,
           int in_num_col_dims)
      : TestCase(place, alias),
        dims_(std::move(dim_in)),
        wdims_(std::move(dim_w)),
        bdims_(dim_b),
        in_num_col_dims_(in_num_col_dims) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto w = scope->FindTensor(weight_);
    auto b = scope->FindTensor(bias_);
    bool flag_bias = b;
    auto out = scope->NewTensor(out_);
    CHECK(out);
    DDim out_dim = compute_out_dim(x->dims(), w->dims(), in_num_col_dims_);
    out->Resize(out_dim);

    LOG(INFO) << "out dims: " << out_dim;

    auto x_data = x->data<float>();
    auto w_data = w->data<float>();
    const float* b_data = nullptr;
    if (flag_bias) {
      b_data = b->data<float>();
    }
    auto out_data = out->mutable_data<float>();

    int m = x->dims().count(0, in_num_col_dims_);
    CHECK_EQ(wdims_[0], x->dims().count(in_num_col_dims_, x->dims().size()));
    int k = wdims_[0];
    int n = wdims_[1];

    LOG(INFO) << "m: " << m << ", n: " << n << ", k: " << k;

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
                 flag_bias,
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
        fill_bias_fc(out_data, b_data, m, n);
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fc");
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("W", {weight_});
    if (bdims_.production() > 0) {
      op_desc->SetInput("Bias", {bias_});
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr<int>("in_num_col_dims", in_num_col_dims_);
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
    SetCommonTensor(weight_, wdims_, win.data());
    if (flag_bias) {
      SetCommonTensor(bias_, bdims_, bin.data());
    }
  }
};

void test_fc(Place place) {
  for (auto& m : {1, 3, 16}) {
    for (auto& n : {1, 4, 16, 128, 256, 1024}) {
      for (auto& k : {1, 16, 128, 1024}) {
        for (auto& bflag : {false, true}) {
          DDim dim_in{{m, k}};
          DDim wdim{{k, n}};
          DDim bdim{{bflag ? n : 0}};
          std::unique_ptr<arena::TestCase> tester(
              new FcOPTest(place, "def", dim_in, wdim, bdim, 1));
#ifdef LITE_WITH_ARM
          auto& ctx = tester->context()->As<ARMContext>();
          ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 1);
#endif
          arena::Arena arena(std::move(tester), place, 6e-5);
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

TEST(FcOP, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_fc(place);
#endif
}

}  // namespace lite
}  // namespace paddle
