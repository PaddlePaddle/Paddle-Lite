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
#include <algorithm>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

template <typename dtype>
void elementwise_compute_ref(const dtype *x_data,
                             const dtype *y_data,
                             dtype *out_data,
                             const DDim &x_dims,
                             const DDim &y_dims,
                             int axis,
                             const std::string elt_type,
                             bool use_relu = false) {
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  // do elementwise add/sub/max/...
  if (elt_type == "add") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype *din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype *dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr + diny_data;
          if (use_relu) {
            *dout_ptr = std::max(*dout_ptr, static_cast<dtype>(0));
          }
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "sub") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype *din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype *dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr - diny_data;
          if (use_relu) {
            *dout_ptr = std::max(*dout_ptr, static_cast<dtype>(0));
          }
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else {
    LOG(FATAL) << "unsupported Elementwise type: " << elt_type << std::endl;
  }
}

TEST(elementwise_add, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "elementwise_add", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";

  lite::Tensor x, y, out;
  operators::ElementwiseParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> ele_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(ele_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(ele_context));

  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 2, 1, 5});
  const DDim y_dim = DDim(std::vector<DDim::value_type>{2, 1, 5});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{3, 2, 1, 5});
  x.Resize(x_dim);
  y.Resize(y_dim);
  out.Resize(out_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }
  auto *mapped_y = static_cast<float *>(
      TargetWrapperCL::Map(y_data, 0, sizeof(float) * y_dim.production()));
  for (int i = 0; i < y_dim.production(); i++) {
    mapped_y[i] = dist(engine);
  }

  kernel->Launch();

  auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto *out_ptr = param.Out->data<float, cl::Buffer>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto &event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  elementwise_compute_ref<float>(
      mapped_x, mapped_y, out_ref.get(), x_dim, y_dim, param.axis, "add");

  TargetWrapperCL::Unmap(x_data, mapped_x);
  TargetWrapperCL::Unmap(y_data, mapped_y);
  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));
  for (int i = 0; i < out_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
}

TEST(fusion_elementwise_add_activation, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels =
      KernelRegistry::Global().Create("fusion_elementwise_add_activation",
                                      TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";

  lite::Tensor x, y, out;
  operators::FusionElementwiseActivationParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;
  param.act_type = "relu";

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> ele_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(ele_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(ele_context));

  const DDim x_dim = DDim(std::vector<DDim::value_type>{30, 20, 10, 50});
  const DDim y_dim = DDim(std::vector<DDim::value_type>{20, 10, 50});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{30, 20, 10, 50});
  x.Resize(x_dim);
  y.Resize(y_dim);
  out.Resize(out_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }
  auto *mapped_y = static_cast<float *>(
      TargetWrapperCL::Map(y_data, 0, sizeof(float) * y_dim.production()));
  for (int i = 0; i < y_dim.production(); i++) {
    mapped_y[i] = dist(engine);
  }

  kernel->Launch();

  auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto *out_ptr = param.Out->data<float, cl::Buffer>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto &event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  elementwise_compute_ref<float>(
      mapped_x, mapped_y, out_ref.get(), x_dim, y_dim, param.axis, "add", true);

  TargetWrapperCL::Unmap(x_data, mapped_x);
  TargetWrapperCL::Unmap(y_data, mapped_y);
  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));
  for (int i = 0; i < out_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_add, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kOpenCL, kFloat, kNCHW, def);
