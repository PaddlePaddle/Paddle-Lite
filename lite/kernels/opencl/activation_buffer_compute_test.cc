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
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {

template <typename dtype>
void relu_compute_ref(const dtype *x_data,
                      const DDim &x_dim,
                      dtype *out_data,
                      float threshold = 0.f) {
  if (abs(threshold) < 1e-5) {
    // relu
    for (int i = 0; i < x_dim.production(); ++i) {
      out_data[i] = (x_data[i] > threshold) ? x_data[i] : threshold;
    }
  } else {
    // relu6 or relu with threshold
    for (int i = 0; i < x_dim.production(); ++i) {
      auto out_tmp = (x_data[i] > 0) ? x_data[i] : 0;
      out_data[i] = (out_tmp < threshold) ? out_tmp : threshold;
    }
  }
}

template <typename dtype>
void sigmoid_compute_ref(const dtype *x_data,
                         const DDim &x_dim,
                         dtype *out_data) {
  for (int i = 0; i < x_dim.production(); ++i) {
    out_data[i] = 1 / (1 + expf(-x_data[i]));
  }
}

TEST(opencl_relu_buffer, compute) {
  // prepare data
  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 6, 10, 10});
  lite::Tensor x, out;
  x.Resize(x_dim);
  out.Resize(x_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }

  // set param and kernel, then run
  operators::ActivationParam param;
  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "relu", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> relu_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(relu_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(relu_context));

  kernel->Launch();

  auto *out_ptr = param.Out->data<float, cl::Buffer>();

  CLRuntime::Global()->command_queue().finish();

  // run compute ref and check
  std::unique_ptr<float[]> out_ref(new float[x_dim.production()]);
  relu_compute_ref<float>(mapped_x, x_dim, out_ref.get());

  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x_data, mapped_x);
}

TEST(opencl_sigmoid_buffer, compute) {
  // prepare data
  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 6, 10, 10});
  lite::Tensor x, out;
  x.Resize(x_dim);
  out.Resize(x_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }

  // set param and kernel, then run
  operators::ActivationParam param;
  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "sigmoid", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> sigmoid_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(sigmoid_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(sigmoid_context));

  kernel->Launch();

  auto *out_ptr = param.Out->data<float, cl::Buffer>();

  CLRuntime::Global()->command_queue().finish();

  // run compute ref and check
  std::unique_ptr<float[]> out_ref(new float[x_dim.production()]);
  sigmoid_compute_ref<float>(mapped_x, x_dim, out_ref.get());

  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x_data, mapped_x);
}

}  // namespace lite
}  // namespace paddle

// sigmoid buffer
USE_LITE_KERNEL(sigmoid, kOpenCL, kFloat, kNCHW, def);

// relu buffer
USE_LITE_KERNEL(relu, kOpenCL, kFloat, kNCHW, def);
