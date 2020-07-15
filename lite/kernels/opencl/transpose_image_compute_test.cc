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
#include "lite/kernels/opencl/test_helper.h"
#include "lite/operators/reshape_op.h"
#include "lite/utils/logging.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

static inline void TestWithKernel(
    const std::unique_ptr<paddle::lite::KernelBase>& kernel) {
  int64_t batch_size = 1;
  int64_t ic = 2;
  int64_t ih = 3;
  int64_t iw = 4;

  int64_t oc = 3;
  int64_t oh = 4;
  int64_t ow = 2;

  lite::Tensor input, output;
  operators::TransposeParam param;

  param.x = &input;
  param.output = &output;
  param.axis = std::vector<int>({0, 2, 3, 1});
  const DDim input_dim =
      lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};
  input.Resize(input_dim);
  const DDim output_dim =
      lite::DDim{std::vector<int64_t>({batch_size, oc, oh, ow})};
  param.output->Resize(output_dim);

  LOG(INFO) << "prepare kernel SetParam------";
  kernel->SetParam(param);

  size_t input_image_width = iw * ((ic + 3) / 4);
  size_t input_image_height = ih * batch_size;

  size_t output_image_width = ow * ((oc + 3) / 4);
  size_t output_image_height = oh * batch_size;

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};

  std::vector<float> input_v(batch_size * ic * ih * iw);

  LOG(INFO) << "gen input ...";

  float* input_v_data = &input_v[0];
  auto index = 0;
  for (auto& i : input_v) {
    i = index++;
  }

  paddle::lite::CLImageConverterDefault default_convertor;

  std::vector<half_t> x_image_data(input_image_width * input_image_height *
                                   4);  // 4 : RGBA

  LOG(INFO) << "set mapped input  ...";
  default_convertor.NCHWToImage(input_v_data, x_image_data.data(), input_dim);

  auto* input_image = input.mutable_data<half_t, cl::Image2D>(
      input_image_width, input_image_height, x_image_data.data());

  LOG(INFO) << "prepare kernel ready";

  LOG(INFO) << "mutable output ...";
  CLImageConverterDefault default_converter;
  DDim out_image_shape = default_converter.InitImageDimInfoWith(output_dim);
  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  auto* out_image = output.mutable_data<half_t, cl::Image2D>(
      out_image_shape[0], out_image_shape[1]);

  LOG(INFO) << "kernel context ...";
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  std::unique_ptr<KernelContext> transpose_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(transpose_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(transpose_context));

  LOG(INFO) << "kernel launch ...";
  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  half_t* out_image_data = new half_t[out_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(out_image_data,
                              output.data<half_t, cl::Image2D>(),
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  float* out_data = new float[out_image_shape.production() * 4];
  default_converter.ImageToNCHW(
      out_image_data, out_data, out_image_shape, output_dim);

  // check output data
  index = 0;
  auto hxw = ih * iw;
  auto cxhxw = ic * hxw;
  for (auto n = 0; n < batch_size; n++) {
    for (auto h = 0; h < ih; h++) {
      for (auto w = 0; w < iw; w++) {
        for (auto c = 0; c < ic; c++) {
          auto input_index = n * cxhxw + c * hxw + h * iw + w;
          auto input_value = input_v_data[input_index];
          auto output_value = out_data[index];
          auto abs_diff = abs(input_value - output_value);
          auto relative_diff = COMPUTE_RELATIVE_DIFF(input_value, output_value);
          EXPECT_EQ(
              (relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
          index++;
        }
      }
    }
  }
}

TEST(transpose_opencl, compute) {
  auto kernels = KernelRegistry::Global().Create("transpose",
                                                 TARGET(kOpenCL),
                                                 PRECISION(kFP16),
                                                 DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  TestWithKernel(kernel);
}

TEST(transpose2_opencl, compute) {
  auto kernels = KernelRegistry::Global().Create("transpose2",
                                                 TARGET(kOpenCL),
                                                 PRECISION(kFP16),
                                                 DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  TestWithKernel(kernel);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(transpose, kOpenCL, kFP16, kImageDefault, image2d);
