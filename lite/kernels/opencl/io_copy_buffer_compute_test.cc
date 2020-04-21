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
#include <iterator>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

TEST(io_copy, compute) {
  LOG(INFO) << "to get kernel ...";
  auto h2d_kernels = KernelRegistry::Global().Create(
      "io_copy", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny));
  ASSERT_FALSE(h2d_kernels.empty());

  auto h2d_kernel = std::move(h2d_kernels.front());
  auto d2h_kernel = std::move(*std::next(h2d_kernels.begin(), 1));
  LOG(INFO) << "get first kernel: " << h2d_kernel->doc();
  LOG(INFO) << "get second kernel: " << d2h_kernel->doc();
  lite::Tensor h_x, d_y, h_y;
  operators::IoCopyParam h2d_param;
  h2d_param.x = &h_x;
  h2d_param.y = &d_y;
  operators::IoCopyParam d2h_param;
  d2h_param.x = &d_y;
  d2h_param.y = &h_y;
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  h2d_kernel->SetParam(h2d_param);
  std::unique_ptr<KernelContext> h2d_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(h2d_context->As<OpenCLContext>()));
  h2d_kernel->SetContext(std::move(h2d_context));
  d2h_kernel->SetParam(d2h_param);
  std::unique_ptr<KernelContext> d2h_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(d2h_context->As<OpenCLContext>()));
  d2h_kernel->SetContext(std::move(d2h_context));

  const DDim dim = DDim(std::vector<DDim::value_type>{3, 9, 28, 28});
  h_x.Resize(dim);
  d_y.Resize(dim);
  h_y.Resize(dim);

  auto* h_x_data = h_x.mutable_data<float>(TARGET(kARM));
  for (int i = 0; i < 3 * 9 * 28 * 28; i++) {
    h_x_data[i] = 3.14f * i / 1000.f;
  }

  h2d_kernel->Launch();
  auto* event_key = d_y.data<float, cl::Buffer>();
  d2h_kernel->Launch();
  auto* h_y_data = h_y.data<float>();

  for (int i = 0; i < 3 * 9 * 28 * 28; i++) {
    EXPECT_NEAR(h_x_data[i], h_y_data[i], 1e-6);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(io_copy, kOpenCL, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kOpenCL, kAny, kAny, device_to_host);
