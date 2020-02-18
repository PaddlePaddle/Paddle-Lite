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
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

void scale(const float* input_data,
           const DDim& in_dim,
           float* output_data,
           const float scale,
           const float bias) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = input_data[i] * scale + bias;
  }
}

TEST(scale_image2d_fp32, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "scale", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel:" << kernel->doc();

  lite::Tensor x, out;
  operators::ScaleParam param;
  param.x = &x;
  param.output = &out;
  param.scale = 1.5f;
  param.bias = 0.3f;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> scale_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(scale_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(scale_context));

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 11, 107, 107});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 11, 107, 107});
  x.Resize(in_dim);
  out.Resize(out_dim);

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);
  std::vector<float> input_v(4 * 11 * 107 * 107);
  for (auto& i : input_v) {
    i = dist(engine);
  }

  LOG(INFO) << "prepare input";
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim image_shape = default_converter->InitImageDimInfoWith(in_dim);
  LOG(INFO) << "image_shape = " << image_shape[0] << " " << image_shape[1];
  std::vector<float> x_image_data(image_shape.production() * 4);  // 4 : RGBA
  default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
  auto* x_image = x.mutable_data<float, cl::Image2D>(
      image_shape[0], image_shape[1], x_image_data.data());
  LOG(INFO) << "x_image:" << x_image;

  auto* out_image =
      out.mutable_data<float, cl::Image2D>(image_shape[0], image_shape[1]);
  LOG(INFO) << "out_image:" << out_image;
  kernel->Launch();

  auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto* out_ptr = param.output->data<float, cl::Image2D>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto& event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  scale(input_v.data(), in_dim, out_ref.get(), 1.5f, 0.3f);

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  float* out_image_data = new float[image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(out_image_data,
                              out_image,
                              image_shape[0],
                              image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  float* out_data = new float[image_shape.production() * 4];
  default_converter->ImageToNCHW(
      out_image_data, out_data, image_shape, out_dim);

  for (int i = 0; i < out_dim.production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref[i], 1e-6);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(scale, kOpenCL, kFloat, kImageDefault, image2d);
