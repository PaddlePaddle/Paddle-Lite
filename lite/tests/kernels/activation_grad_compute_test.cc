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

#include "lite/kernels/host/activation_grad_compute.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/activation_compute.h"
#include "lite/kernels/arm/activation_extra_compute.h"

namespace paddle {
namespace lite {
namespace kernels {

using param_t = operators::ActivationParam;
using grad_param_t = operators::ActivationGradParam;

template <class kernel_t, class grad_kernel_t>
class ActivationGradTester {
 public:
  explicit ActivationGradTester(DDim dims) : dims_(dims) {}

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
                   const std::vector<float>& in_vec,
                   float* out_vec) {
    Tensor x;
    Tensor output;
    x.Resize(dims_);
    output.Resize(dims_);
    auto* x_data = x.mutable_data<float>();
    for (int i = 0; i < dims_.production(); i++) {
      x_data[i] = in_vec[i];
    }
    param->X = &x;
    param->Out = &output;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* output_data = output.mutable_data<float>();
    for (int i = 0; i < dims_.production(); i++) {
      out_vec[i] = output_data[i];
    }
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_t* kernel,
                    const std::vector<float>& in_vec,
                    const std::vector<float>& out_vec,
                    const std::vector<float>& out_grad_vec,
                    float* in_grad_vec) {
    Tensor x;
    Tensor out;
    Tensor x_grad;
    Tensor out_grad;
    x.Resize(dims_);
    out.Resize(dims_);
    x_grad.Resize(dims_);
    out_grad.Resize(dims_);
    auto* x_data = x.mutable_data<float>();
    auto* out_data = out.mutable_data<float>();
    auto* out_grad_data = out_grad.mutable_data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      x_data[i] = in_vec[i];
      out_data[i] = out_vec[i];
      out_grad_data[i] = out_grad_vec[i];
    }
    param->X = &x;
    param->Out = &out;
    param->X_grad = &x_grad;
    param->Out_grad = &out_grad;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* x_grad_data = x_grad.mutable_data<float>();
    for (int i = 0; i < dims_.production(); i++) {
      in_grad_vec[i] = x_grad_data[i];
    }
  }

  void check_grad(float delta, float max_grad_delta) {
    std::vector<float> x(dims_.production());
    std::vector<float> out(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      x[i] = static_cast<float>(i % 3 - 2.0) / 2.0 * 0.333 +
             static_cast<float>(i % 19 - 10.0) / 10.0 * 0.333 +
             static_cast<float>(i % 39 - 20.0) / 20.0 * 0.333 + 0.001213;
    }
    this->run_forward(&param_, &kernel_, x, out.data());

    std::vector<float> x_delta(dims_.production());
    std::vector<float> out_delta(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      x_delta[i] = x[i] + delta;
    }
    this->run_forward(&delta_param_, &delta_kernel_, x_delta, out_delta.data());

    std::vector<float> out_grad(dims_.production());
    std::vector<float> x_grad(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      out_grad[i] = 1.0;
    }
    this->run_backward(
        &grad_param_, &grad_kernel_, x, out, out_grad, x_grad.data());

    for (int i = 0; i < dims_.production(); i++) {
      EXPECT_NEAR(x_grad[i], (out_delta[i] - out[i]) / delta, max_grad_delta);
    }
  }

 private:
  DDim dims_;
  kernel_t kernel_;
  kernel_t delta_kernel_;
  grad_kernel_t grad_kernel_;
  param_t param_;
  param_t delta_param_;
  grad_param_t grad_param_;
};

void TestSquareGrad(DDim dims) {
  LOG(INFO) << "Test Square grad";
  std::unique_ptr<
      ActivationGradTester<arm::SquareCompute, host::SquareGradCompute>>
      tester(
          new ActivationGradTester<arm::SquareCompute, host::SquareGradCompute>(
              dims));
  tester->prepare_kernel();
  float delta = 0.001;
  float max_grad_delta = 0.005;
  tester->check_grad(delta, max_grad_delta);
}

void TestReluGrad(DDim dims) {
  LOG(INFO) << "Test Relu grad";
  std::unique_ptr<ActivationGradTester<arm::ReluCompute<PRECISION(kFloat)>,
                                       host::ReluGradCompute>>
      tester(new ActivationGradTester<arm::ReluCompute<PRECISION(kFloat)>,
                                      host::ReluGradCompute>(dims));
  tester->prepare_kernel();
  float delta = 0.001;
  float max_grad_delta = 0.005;
  tester->check_grad(delta, max_grad_delta);
}

void TestTanhGrad(DDim dims) {
  LOG(INFO) << "Test Tanh grad";
  std::unique_ptr<ActivationGradTester<arm::TanhCompute<PRECISION(kFloat)>,
                                       host::TanhGradCompute>>
      tester(new ActivationGradTester<arm::TanhCompute<PRECISION(kFloat)>,
                                      host::TanhGradCompute>(dims));
  tester->prepare_kernel();
  float delta = 0.001;
  float max_grad_delta = 0.005;
  tester->check_grad(delta, max_grad_delta);
}

TEST(activation_grad_host, compute) {
  DeviceInfo::Init();
  for (auto n : {2, 1}) {
    for (auto c : {2, 9}) {
      for (auto h : {2, 1}) {
        for (auto w : {2, 10}) {
          TestSquareGrad(DDim(std::vector<int64_t>({n, c, h, w})));
          TestReluGrad(DDim(std::vector<int64_t>({n, c, h, w})));
          TestTanhGrad(DDim(std::vector<int64_t>({n, c, h, w})));
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(square, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(square_grad, kHost, kFloat, kNCHW, def);
