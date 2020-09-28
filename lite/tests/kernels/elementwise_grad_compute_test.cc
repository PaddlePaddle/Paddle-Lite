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

#include "lite/kernels/arm/elementwise_grad_compute.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/elementwise_compute.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using param_t = operators::ElementwiseParam;
using grad_param_t = operators::ElementwiseGradParam;
using kernel_add_t = ElementwiseAddCompute<float, PRECISION(kFloat)>;
using grad_kernel_add_t = ElementwiseAddGradCompute;
using kernel_sub_t = ElementwiseSubCompute<float, PRECISION(kFloat)>;
using grad_kernel_sub_t = ElementwiseSubGradCompute;

void elementwise_common(grad_param_t& param,           // NOLINT
                        std::vector<float>& out_grad,  // NOLINT
                        std::vector<float>& x_grad,    // NOLINT
                        std::vector<float>& y_grad,    // NOLINT
                        std::string flag) {
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  if (x_dims == y_dims) {
    for (int i = 0; i < x_dims.production(); ++i) {
      if (flag == "add") {
        x_grad[i] = out_grad[i];
        y_grad[i] = out_grad[i];
      }
      if (flag == "sub") {
        x_grad[i] = out_grad[i];
        y_grad[i] = -out_grad[i];
      }
    }
  } else {
    LOG(FATAL) << "unsupport dims";
  }
}

class ElementwiseAddGradTester {
 public:
  explicit ElementwiseAddGradTester(const DDim& x_dims,
                                    const DDim& y_dims,
                                    int axis)
      : x_dims_(x_dims), y_dims_(y_dims), axis_(axis) {}

  void prepare_kernel() {
    std::unique_ptr<KernelContext> ctx1(new KernelContext);
    ctx1->As<ARMContext>();
    kernel_.SetContext(std::move(ctx1));

    std::unique_ptr<KernelContext> ctx3(new KernelContext);
    ctx3->As<ARMContext>();
    grad_kernel_.SetContext(std::move(ctx3));
  }

  void run_forward(param_t* param,
                   kernel_add_t* kernel,
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

    param->X = &x;
    param->Y = &y;
    param->Out = &output;
    param->axis = axis_;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* output_data = output.mutable_data<float>();
    for (int i = 0; i < out_dims_.production(); i++) {
      out_vec[i] = output_data[i];
    }
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_add_t* kernel,
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

    param->X = &x;
    param->XGrad = &x_grad;
    param->Y = &y;
    param->YGrad = &y_grad;
    param->OutGrad = &out_grad;
    param->axis = axis_;

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

  void check_grad(float delta2, float max_grad_delta2) {
    std::vector<int64_t> out_shape;
    // infer shape
    auto x_dim = x_dims_;
    auto y_dim = y_dims_;
    if (x_dim == y_dim) {
      out_dims_ = x_dim;
    } else {
      int max_dim = (x_dim.size() > y_dim.size() ? x_dim.size() : y_dim.size());
      int axis = param_.axis;
      axis =
          (axis == -1 ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                      : axis);
      std::vector<int64_t> x_dims_array(max_dim);
      std::vector<int64_t> y_dims_array(max_dim);
      std::vector<int64_t> out_dims_array(max_dim);

      if (x_dim.size() > y_dim.size()) {
        for (int i = 0; i < axis; ++i) {
          y_dims_array[i] = 1;
        }
        if (axis + y_dim.size() < max_dim) {
          for (int i = axis + y_dim.size(); i < max_dim; ++i) {
            y_dims_array[i] = 1;
          }
        }
        x_dims_array = x_dim.Vectorize();
        for (int i = 0; i < y_dim.size(); ++i) {
          y_dims_array[i + axis] = y_dim[i];
        }
      } else {
        for (int i = 0; i < axis; ++i) {
          x_dims_array[i] = 1;
        }
        if (axis + x_dim.size() < max_dim) {
          for (int i = axis + x_dim.size(); i < max_dim; ++i) {
            x_dims_array[i] = 1;
          }
        }
        y_dims_array = y_dim.Vectorize();
        for (int i = 0; i < x_dim.size(); ++i) {
          x_dims_array[i + axis] = x_dim[i];
        }
      }
      for (int i = 0; i < max_dim; i++) {
        if (x_dims_array[i] == -1 || y_dims_array[i] == -1) {
          out_dims_array[i] = -1;
        } else {
          out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
        }
      }
      out_dims_ = DDim(out_dims_array);
    }
    // infer end
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
    Tensor tensor_x;
    Tensor tensor_y;
    tensor_x.Resize(x_dims_);
    tensor_y.Resize(y_dims_);
    grad_param_.X = &tensor_x;
    grad_param_.Y = &tensor_y;

    elementwise_common(grad_param_, out_grad, x_delta, y_delta, "add");

    float max_grad_delta = 0.0005;
    for (int i = 0; i < x_dims_.production(); i++) {
      EXPECT_NEAR(x_grad[i], x_delta[i], max_grad_delta);
      EXPECT_NEAR(y_grad[i], y_delta[i], max_grad_delta);
    }
  }

 private:
  DDim x_dims_;
  DDim y_dims_;
  DDim out_dims_;
  int axis_;
  kernel_add_t kernel_;
  grad_kernel_add_t grad_kernel_;
  param_t param_;
  grad_param_t grad_param_;
};

class ElementwiseSubGradTester {
 public:
  explicit ElementwiseSubGradTester(const DDim& x_dims,
                                    const DDim& y_dims,
                                    int axis)
      : x_dims_(x_dims), y_dims_(y_dims), axis_(axis) {}

  void prepare_kernel() {
    std::unique_ptr<KernelContext> ctx1(new KernelContext);
    ctx1->As<ARMContext>();
    kernel_.SetContext(std::move(ctx1));

    std::unique_ptr<KernelContext> ctx3(new KernelContext);
    ctx3->As<ARMContext>();
    grad_kernel_.SetContext(std::move(ctx3));
  }

  void run_forward(param_t* param,
                   kernel_sub_t* kernel,
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

    param->X = &x;
    param->Y = &y;
    param->Out = &output;
    param->axis = axis_;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* output_data = output.mutable_data<float>();
    for (int i = 0; i < out_dims_.production(); i++) {
      out_vec[i] = output_data[i];
    }
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_sub_t* kernel,
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

    param->X = &x;
    param->XGrad = &x_grad;
    param->Y = &y;
    param->YGrad = &y_grad;
    param->OutGrad = &out_grad;
    param->axis = axis_;

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

  void check_grad(float delta2, float max_grad_delta2) {
    std::vector<int64_t> out_shape;
    // infer shape
    auto x_dim = x_dims_;
    auto y_dim = y_dims_;
    if (x_dim == y_dim) {
      out_dims_ = x_dim;
    } else {
      int max_dim = (x_dim.size() > y_dim.size() ? x_dim.size() : y_dim.size());
      int axis = param_.axis;
      axis =
          (axis == -1 ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                      : axis);
      std::vector<int64_t> x_dims_array(max_dim);
      std::vector<int64_t> y_dims_array(max_dim);
      std::vector<int64_t> out_dims_array(max_dim);

      if (x_dim.size() > y_dim.size()) {
        for (int i = 0; i < axis; ++i) {
          y_dims_array[i] = 1;
        }
        if (axis + y_dim.size() < max_dim) {
          for (int i = axis + y_dim.size(); i < max_dim; ++i) {
            y_dims_array[i] = 1;
          }
        }
        x_dims_array = x_dim.Vectorize();
        for (int i = 0; i < y_dim.size(); ++i) {
          y_dims_array[i + axis] = y_dim[i];
        }
      } else {
        for (int i = 0; i < axis; ++i) {
          x_dims_array[i] = 1;
        }
        if (axis + x_dim.size() < max_dim) {
          for (int i = axis + x_dim.size(); i < max_dim; ++i) {
            x_dims_array[i] = 1;
          }
        }
        y_dims_array = y_dim.Vectorize();
        for (int i = 0; i < x_dim.size(); ++i) {
          x_dims_array[i + axis] = x_dim[i];
        }
      }
      for (int i = 0; i < max_dim; i++) {
        if (x_dims_array[i] == -1 || y_dims_array[i] == -1) {
          out_dims_array[i] = -1;
        } else {
          out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
        }
      }
      out_dims_ = DDim(out_dims_array);
    }
    // infer end
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
    Tensor tensor_x;
    Tensor tensor_y;
    tensor_x.Resize(x_dims_);
    tensor_y.Resize(y_dims_);
    grad_param_.X = &tensor_x;
    grad_param_.Y = &tensor_y;

    elementwise_common(grad_param_, out_grad, x_delta, y_delta, "sub");

    float max_grad_delta = 0.0005;
    for (int i = 0; i < x_dims_.production(); i++) {
      EXPECT_NEAR(x_grad[i], x_delta[i], max_grad_delta);
      EXPECT_NEAR(y_grad[i], y_delta[i], max_grad_delta);
    }
  }

 private:
  DDim x_dims_;
  DDim y_dims_;
  DDim out_dims_;
  int axis_;
  kernel_sub_t kernel_;
  grad_kernel_sub_t grad_kernel_;
  param_t param_;
  grad_param_t grad_param_;
};
void TestNormalCase(const std::vector<int64_t>& x_dims,
                    const std::vector<int64_t>& y_dims,
                    int axis) {
  std::unique_ptr<ElementwiseAddGradTester> tester_add(
      new ElementwiseAddGradTester(DDim(x_dims), DDim(y_dims), axis));
  std::unique_ptr<ElementwiseSubGradTester> tester_sub(
      new ElementwiseSubGradTester(DDim(x_dims), DDim(y_dims), axis));

  tester_add->prepare_kernel();
  tester_sub->prepare_kernel();
  float delta = 0.001;
  float max_grad_delta = 0.005;
  tester_add->check_grad(delta, max_grad_delta);
  tester_sub->check_grad(delta, max_grad_delta);
}

TEST(mul_grad_arm, compute) {
  LOG(INFO) << "Test Elementwise grad";
  DeviceInfo::Init();
  TestNormalCase({3, 2}, {3, 2}, 0);
  TestNormalCase({3, 5}, {3, 5}, 1);
  TestNormalCase({3, 4, 3}, {3, 4, 3}, 0);
  TestNormalCase({9, 2, 5}, {9, 2, 5}, 1);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(elementwise_add_grad, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
