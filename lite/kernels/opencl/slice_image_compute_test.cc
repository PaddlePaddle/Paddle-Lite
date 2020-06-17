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
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {

void slice_channel(const float* input_data,
                   const DDim& in_dim,
                   float* output_data,
                   const int start,
                   const int end) {
  int n = in_dim[0];
  int in_n_stride = 1;
  for (int i = 1; i < in_dim.size(); ++i) {
    in_n_stride *= in_dim[i];
  }
  int in_c_stride = in_n_stride / in_dim[1];
  int mini_batch = end - start;
  for (int ni = 0; ni < n; ++ni) {
    const float* in_n = input_data + ni * in_n_stride + start * in_c_stride;
    float* out_n = output_data + ni * mini_batch * in_c_stride;
    memcpy(out_n, in_n, sizeof(float) * mini_batch * in_c_stride);
  }
}

TEST(slice_image2d_fp16, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "slice", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel:" << kernel->doc();

  lite::Tensor x, out;
  operators::SliceParam param;
  param.X = &x;
  param.Out = &out;
  param.axes = std::vector<int>({1});
  param.starts = std::vector<int32_t>({2});
  param.ends = std::vector<int32_t>({5});

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> slice_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(slice_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(slice_context));

  const DDim in_dim = DDim(std::vector<DDim::value_type>{3, 11, 107, 218});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{3, 3, 107, 218});
  x.Resize(in_dim);
  out.Resize(out_dim);

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);
  std::vector<float> input_v(3 * 11 * 107 * 218);
  for (auto& i : input_v) {
    i = dist(engine);
  }

  LOG(INFO) << "prepare input";
  std::unique_ptr<CLImageConverterDefault> default_converter(
      new CLImageConverterDefault());
  DDim image_shape = default_converter->InitImageDimInfoWith(in_dim);
  LOG(INFO) << "image_shape = " << image_shape[0] << " " << image_shape[1];
  std::vector<half_t> x_image_data(image_shape.production() * 4);  // 4 : RGBA
  default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
  auto* x_image = x.mutable_data<half_t, cl::Image2D>(
      image_shape[0], image_shape[1], x_image_data.data());
  LOG(INFO) << "x_image:" << x_image;

  auto* out_image =
      out.mutable_data<half_t, cl::Image2D>(image_shape[0], image_shape[1]);
  LOG(INFO) << "out_image:" << out_image;
  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  slice_channel(input_v.data(), in_dim, out_ref.get(), 2, 5);

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  half_t* out_image_data = new half_t[image_shape.production() * 4];
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
    auto abs_diff = abs(out_data[i] - out_ref[i]);
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
    EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
    if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
      LOG(ERROR) << "error idx:" << i << " out_data[" << i
                 << "]:" << out_data[i] << " "
                                           "out_ref["
                 << i << "]:" << out_ref[i] << " abs_diff:" << abs_diff
                 << " relative_diff:" << relative_diff
                 << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(slice, kOpenCL, kFP16, kImageDefault, image2d);
