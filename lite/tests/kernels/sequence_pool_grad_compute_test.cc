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

#include "lite/kernels/arm/sequence_pool_grad_compute.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/sequence_pool_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using param_t = operators::SequencePoolParam;
using grad_param_t = operators::SequencePoolGradParam;
using kernel_t = SequencePoolCompute;
using grad_kernel_t = SequencePoolGradCompute;

class SequencePoolGradTester {
 public:
  explicit SequencePoolGradTester(DDim dims,
                                  std::vector<std::vector<uint64_t>> lod,
                                  std::string pool_type)
      : dims_(dims), lod_(lod), pool_type_(pool_type) {}

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
    x.set_lod(lod_);
    param->X = &x;
    param->pool_type = pool_type_;
    param->Out = &output;
    kernel->SetParam(*param);
    kernel->Launch();
    auto* output_data = output.mutable_data<float>();
    for (int i = 0; i < output.numel(); i++) {
      out_vec[i] = output_data[i];
    }
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_t* kernel,
                    const std::vector<float>& in_vec,
                    const std::vector<float>& out_grad_vec,
                    float* in_grad_vec) {
    Tensor x;
    Tensor x_grad;
    Tensor out_grad;
    x.Resize(dims_);
    x_grad.Resize(dims_);
    x.set_lod(lod_);
    // backword
    out_grad.Resize(out_dims_);
    auto* x_data = x.mutable_data<float>();
    auto* out_grad_data = out_grad.mutable_data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      x_data[i] = in_vec[i];
    }
    for (int i = 0; i < out_dims_.production(); i++) {
      out_grad_data[i] = out_grad_vec[i];
    }
    param->X = &x;
    param->X_Grad = &x_grad;
    param->Out_Grad = &out_grad;
    param->pool_type = pool_type_;
    kernel->SetParam(*param);
    kernel->Launch();
    auto* x_grad_data = x_grad.data<float>();
    for (int i = 0; i < dims_.production(); i++) {
      in_grad_vec[i] = x_grad_data[i];
    }
  }

  void check_grad(float delta, float max_grad_delta) {
    std::vector<int64_t> out_shape;
    out_dims_ = dims_;
    out_dims_[0] = lod_[0].size() - 1;
    std::vector<float> x(dims_.production());
    std::vector<float> out(out_dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      x[i] = static_cast<float>(i % 3 - 2.0) / 2.0 * 0.333 +
             static_cast<float>(i % 19 - 10.0) / 10.0 * 0.333 +
             static_cast<float>(i % 39 - 20.0) / 20.0 * 0.333 + 0.001213;
    }
    LOG(INFO) << "run_forward:";
    this->run_forward(&param_, &kernel_, x, out.data());

    std::vector<float> out_grad(out_dims_.production());
    std::vector<float> x_grad(dims_.production());
    std::vector<float> x_delta(dims_.production());
    std::vector<float> out_delta(out_dims_.production());
    for (int i = 0; i < out_dims_.production(); i++) {
      out_grad[i] = 1.0;
      x_grad[i] = 1.0;
    }
    LOG(INFO) << "run_backward:";
    this->run_backward(&grad_param_, &grad_kernel_, x, out_grad, x_grad.data());
    LOG(INFO) << "end";
  }

 private:
  DDim dims_;
  DDim out_dims_;
  std::vector<std::vector<uint64_t>> lod_;
  std::string pool_type_;
  kernel_t kernel_;
  kernel_t delta_kernel_;
  grad_kernel_t grad_kernel_;
  param_t param_;
  param_t delta_param_;
  grad_param_t grad_param_;
};

void generate_lod(int seq_num,
                  int max_len,
                  std::vector<uint64_t>& seq_offset) {  // NOLINT
  seq_offset.clear();
  int sum = 0;
  seq_offset.push_back(sum);
  for (int i = 0; i < seq_num; i++) {
    sum += std::rand() % max_len + 1;
    seq_offset.push_back(uint64_t(sum));
  }
}

void TestSequencePoolGrad(DDim dims,
                          std::vector<std::vector<uint64_t>> lod,
                          std::string pool_type) {
  LOG(INFO) << "Test SequencePool grad";
  std::unique_ptr<SequencePoolGradTester> tester(
      new SequencePoolGradTester(dims, lod, pool_type));
  tester->prepare_kernel();
  float delta = 0.001;
  float max_grad_delta = 0.005;
  tester->check_grad(delta, max_grad_delta);
}

TEST(sequence_pool_grad_host, compute) {
#ifdef LITE_WITH_ARM
  int max_len = 2;
  for (auto c : {2, 4}) {
    for (auto h : {1, 3, 4}) {
      for (auto w : {1, 3, 4}) {
        for (auto pool_type :
             {"SUM", "AVERAGE", "SQRT", "MAX", "MIN", "FIRST", "LAST"}) {
          for (auto seq_num : {1, 3, 5}) {
            std::vector<std::vector<uint64_t>> lod;
            lod.resize(1);
            generate_lod(seq_num, max_len, lod[0]);
            int64_t n = int64_t(lod[0].back());
            LOG(INFO) << "sequence_pool_grad parameter: "
                      << ", n = " << n << ", c = " << c << ", h = " << h
                      << ", w = " << w << ", seq_num = " << seq_num
                      << ", pool_type = " << pool_type;
            TestSequencePoolGrad(
                DDim(std::vector<int64_t>({n, c, h, w})), lod, pool_type);
          }
        }
      }
    }
  }
#endif
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(sequence_pool, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_pool_grad, kARM, kFloat, kNCHW, def);
