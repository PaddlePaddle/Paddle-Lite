// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/mul_grad_compute.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/mul_compute.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using param_t = operators::MulParam;
using grad_param_t = operators::MulGradParam;
using kernel_t = MulCompute<PRECISION(kFloat), PRECISION(kFloat)>;
using grad_kernel_t = MulGradCompute;

class MulGradTester {
 public:
  explicit MulGradTester(const DDim& x_dims,
                         const DDim& y_dims,
                         int x_num_col_dims,
                         int y_num_col_dims)
      : x_dims_(x_dims),
        y_dims_(y_dims),
        x_num_col_dims_(x_num_col_dims),
        y_num_col_dims_(y_num_col_dims) {}

  void prepare_kernel() {
    std::unique_ptr<KernelContext> ctx1(new KernelContext);
    ctx1->As<ARMContext>();
    kernel_.SetContext(std::move(ctx1));

    std::unique_ptr<KernelContext> ctx2(new KernelContext);
    ctx2->As<ARMContext>();
    delta_kernel_.SetContext(std::move(ctx2));

    std::unique_ptr<KernelContext> ctx3(new KernelContext);
    ctx3->As<ARMContext>();
    grad_kernel_.SetContext(std::move(ctx3));
  }

  void run_forward(param_t* param,
                   kernel_t* kernel,
                   const std::vector<float>& x_vec,
                   const std::vector<float>& y_vec,
                   float* out_vec) {
    Tensor x;
    Tensor y;
    Tensor output;
    x.Resize(x_dims_);
    y.Resize(y_dims_);
    output.Resize(DDim(out_dims_));
    auto* x_data = x.mutable_data<float>();
    auto* y_data = y.mutable_data<float>();
    for (int i = 0; i < x_dims_.production(); i++) {
      x_data[i] = x_vec[i];
    }
    for (int i = 0; i < y_dims_.production(); i++) {
      y_data[i] = y_vec[i];
    }

    param->x = &x;
    param->y = &y;
    param->output = &output;
    param->x_num_col_dims = x_num_col_dims_;
    param->y_num_col_dims = y_num_col_dims_;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* output_data = output.mutable_data<float>();
    for (int i = 0; i < out_dims_.production(); i++) {
      out_vec[i] = output_data[i];
    }
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_t* kernel,
                    const std::vector<float>& x_vec,
                    const std::vector<float>& y_vec,
                    const std::vector<float>& out_grad_vec,
                    float* x_grad_vec,
                    float* y_grad_vec) {
    Tensor x;
    Tensor x_grad;
    Tensor y;
    Tensor y_grad;
    Tensor out_grad;
    x.Resize(x_dims_);
    x_grad.Resize(x_dims_);
    y.Resize(y_dims_);
    y_grad.Resize(y_dims_);
    out_grad.Resize(out_dims_);
    auto* x_data = x.mutable_data<float>();
    auto* y_data = y.mutable_data<float>();
    auto* out_grad_data = out_grad.mutable_data<float>();
    for (int i = 0; i < x_dims_.production(); i++) {
      x_data[i] = x_vec[i];
    }
    for (int i = 0; i < y_dims_.production(); i++) {
      y_data[i] = y_vec[i];
    }
    for (int i = 0; i < out_dims_.production(); i++) {
      out_grad_data[i] = out_grad_vec[i];
    }

    param->x = &x;
    param->x_grad = &x_grad;
    param->y = &y;
    param->y_grad = &y_grad;
    param->output_grad = &out_grad;
    param->x_num_col_dims = x_num_col_dims_;
    param->y_num_col_dims = y_num_col_dims_;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* x_grad_data = x_grad.mutable_data<float>();
    auto* y_grad_data = y_grad.mutable_data<float>();
    for (int i = 0; i < x_dims_.production(); i++) {
      x_grad_vec[i] = x_grad_data[i];
    }
    for (int i = 0; i < y_dims_.production(); i++) {
      y_grad_vec[i] = y_grad_data[i];
    }
  }

  void check_grad() {
    std::vector<int64_t> out_shape;
    for (int i = 0; i < x_num_col_dims_; i++) {
      out_shape.push_back(x_dims_[i]);
    }
    for (int i = y_num_col_dims_; i < y_dims_.size(); i++) {
      out_shape.push_back(y_dims_[i]);
    }
    out_dims_ = DDim(out_shape);

    // forward
    std::vector<float> x(x_dims_.production());
    std::vector<float> y(y_dims_.production());
    std::vector<float> out(out_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    fill_data_rand(y.data(), -1.f, 1.f, y_dims_.production());
    this->run_forward(&param_, &kernel_, x, y, out.data());

    // backward
    std::vector<float> out_grad(out_dims_.production());
    std::vector<float> x_grad(x_dims_.production());
    std::vector<float> y_grad(y_dims_.production());
    for (int i = 0; i < out_dims_.production(); i++) {
      out_grad[i] = 1.0;
    }
    this->run_backward(&grad_param_,
                       &grad_kernel_,
                       x,
                       y,
                       out_grad,
                       x_grad.data(),
                       y_grad.data());

    // get numeric gradient
    std::vector<float> x_delta(x_dims_.production());
    std::vector<float> y_delta(y_dims_.production());
    std::vector<float> out_delta(out_dims_.production());

    float delta = 0.001;
    float max_grad_delta = 0.0055;
    for (int i = 0; i < x_dims_.production(); i++) {
      for (int j = 0; j < x_dims_.production(); j++) {
        if (i == j) {
          x_delta[j] = x[j] + delta;
        } else {
          x_delta[j] = x[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, x_delta, y, out_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dims_.production(); j++) {
        sum += (out_delta[j] - out[j]);
      }

      EXPECT_NEAR(x_grad[i], sum / delta, max_grad_delta);
    }

    for (int i = 0; i < y_dims_.production(); i++) {
      for (int j = 0; j < y_dims_.production(); j++) {
        y_delta[j] = i == j ? y[j] + delta : y[j];
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, x, y_delta, out_delta.data());
      float sum = 0;
      for (int j = 0; j < out_dims_.production(); j++) {
        sum += out_delta[j] - out[j];
      }

      EXPECT_NEAR(y_grad[i], sum / delta, max_grad_delta);
    }
  }

 private:
  DDim x_dims_;
  DDim y_dims_;
  DDim out_dims_;
  int x_num_col_dims_;
  int y_num_col_dims_;
  kernel_t kernel_;
  kernel_t delta_kernel_;
  grad_kernel_t grad_kernel_;
  param_t param_;
  param_t delta_param_;
  grad_param_t grad_param_;
};

void TestNormalCase(const std::vector<int64_t>& x_dims,
                    const std::vector<int64_t>& y_dims,
                    int x_num_col_dims,
                    int y_num_col_dims) {
  std::unique_ptr<MulGradTester> tester(new MulGradTester(
      DDim(x_dims), DDim(y_dims), x_num_col_dims, y_num_col_dims));

  tester->prepare_kernel();

  tester->check_grad();
}

TEST(mul_grad_arm, compute) {
  LOG(INFO) << "Test Mul grad";
  DeviceInfo::Init();
  TestNormalCase({1, 3}, {3, 2}, 1, 1);
  TestNormalCase({3, 2}, {2, 1}, 1, 1);
  TestNormalCase({3, 1}, {1, 7}, 1, 1);
  TestNormalCase({2, 3}, {3, 2}, 1, 1);
  TestNormalCase({4, 5}, {5, 4}, 1, 1);
  TestNormalCase({4, 5}, {5, 4, 3, 2}, 1, 1);
  TestNormalCase({3, 4}, {2, 2, 3}, 1, 2);
  TestNormalCase({4, 20}, {5, 4, 3, 2}, 1, 2);
  TestNormalCase({4, 60}, {5, 4, 3, 2}, 1, 3);
  TestNormalCase({2, 3, 4, 5}, {60, 4}, 1, 1);
  TestNormalCase({2, 3, 4, 5}, {20, 4}, 2, 1);
  TestNormalCase({2, 3, 4, 5}, {5, 4}, 3, 1);
  TestNormalCase({2, 3, 4, 5}, {60, 3, 4, 5}, 1, 1);
  TestNormalCase({2, 3, 4, 5}, {4, 5, 6, 2}, 2, 2);
  TestNormalCase({2, 3, 4, 5}, {5, 1, 4, 2}, 3, 2);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul_grad, kARM, kFloat, kNCHW, def);
