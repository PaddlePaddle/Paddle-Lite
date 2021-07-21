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

template <typename InT, typename OutT>
void cast_compute_ref(const InT *x_data,
                      const DDim &x_dim,
                      OutT *out_data,
                      float threshold = 0.f) {
  // cast
  for (int i = 0; i < x_dim.production(); ++i) {
    out_data[i] = static_cast<OutT>(x_data[i]);
  }
}

TEST(opencl_cast_fp32_to_int32_buffer, compute) {
  // prepare data
  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 6, 10, 10});
  lite::Tensor x, out;
  x.Resize(x_dim);
  out.Resize(x_dim);

  // set param and kernel, then run
  operators::CastParam param;
  param.X = &x;
  param.Out = &out;
  param.out_dtype = 2;
  param.in_dtype = 5;

  // float to int
  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "cast", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> cast_contest(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(cast_contest->As<OpenCLContext>()));
  kernel->SetContext(std::move(cast_contest));

  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  // run compute ref and check
  std::unique_ptr<int[]> out_ref(new int[x_dim.production()]);
  cast_compute_ref<float, int>(mapped_x, x_dim, out_ref.get());

  auto *out_data = out.mutable_data<int, cl::Buffer>();
  auto *mapped_out = static_cast<int *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(int) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x_data, mapped_x);
}

TEST(opencl_cast_int32_to_fp32_buffer, compute) {
  // prepare data
  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 6, 10, 10});
  lite::Tensor x, out;
  x.Resize(x_dim);
  out.Resize(x_dim);

  // set param and kernel, then run
  operators::CastParam param;
  param.X = &x;
  param.Out = &out;
  param.out_dtype = 5;
  param.in_dtype = 2;

  // float to int
  auto *x_data = x.mutable_data<int, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-100, 100);
  auto *mapped_x = static_cast<int *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(int) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = static_cast<int>(dist(engine));
  }

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "cast", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> cast_contest(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(cast_contest->As<OpenCLContext>()));
  kernel->SetContext(std::move(cast_contest));

  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  // run compute ref and check
  std::unique_ptr<float[]> out_ref(new float[x_dim.production()]);
  cast_compute_ref<int, float>(mapped_x, x_dim, out_ref.get());

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

// cast buffer
USE_LITE_KERNEL(cast, kOpenCL, kFloat, kNCHW, def);
