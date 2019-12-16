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
#include "lite/operators/reshape_op.h"
#include "lite/utils/logging.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
static DDim ValidateShape(const std::vector<int>& shape,
                          const DDim& input_dims) {
  const lite::DDim::value_type input_size = input_dims.production();
  auto input_shape = input_dims.Vectorize();
  bool all_positive = std::all_of(
      input_shape.cbegin(), input_shape.cend(), [](lite::DDim::value_type i) {
        return i > 0;
      });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int unk_dim_val = -1;
  const int copy_dim_val = 0;

  std::vector<lite::DDim::value_type> output_shape(shape.size(), 0);
  lite::DDim::value_type capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(static_cast<int>(i), input_shape.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }

    capacity *= (shape[i] ? static_cast<lite::DDim::value_type>(shape[i])
                          : input_shape[i]);
    output_shape[i] = (shape[i] ? static_cast<lite::DDim::value_type>(shape[i])
                                : input_shape[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -input_size / capacity;
      CHECK_EQ(output_shape[unk_dim_idx] * capacity, -input_size)
          << "Invalid shape is given.";
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, input_size) << "Invalid shape is given.";
  }
  return lite::DDim(output_shape);
}

TEST(reshape_opencl, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "reshape", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  LOG(INFO) << "created reshape kernel";

  LOG(INFO) << "prepare kernel ------";

  int64_t batch_size = 1;
  int64_t ic = 2;
  int64_t ih = 4;
  int64_t iw = 6;

  lite::Tensor input, output;

  operators::ReshapeParam param;

  Tensor shape_tensor;
  shape_tensor.Resize({2});
  auto* shape_tensor_data = shape_tensor.mutable_data<int>();
  shape_tensor_data[0] = 6;
  shape_tensor_data[1] = 8;

  param.x = &input;
  param.shape_tensor = &shape_tensor;  // use shape_tensor
  param.inplace = false;
  param.output = &output;

  const DDim input_dim =
      lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};
  input.Resize(input_dim);

  std::vector<int> final_shape = std::vector<int>(
      shape_tensor_data, shape_tensor_data + shape_tensor.numel());

  auto output_dim = ValidateShape(final_shape, input_dim);
  param.output->Resize(output_dim);
  LOG(INFO) << " output_dim------" << output_dim;

  LOG(INFO) << "prepare kernel SetParam------";
  kernel->SetParam(param);

  size_t input_image_width = iw * ((ic + 3) / 4);
  size_t input_image_height = ih * batch_size;

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};

  //  LOG(INFO) << "map input ...";
  //  auto* mapped_input =
  //      static_cast<float*>(TargetWrapperCL::MapImage(input_data,
  //                                                    input_image_width,
  //                                                    input_image_height,
  //                                                    cl_image2d_row_pitch,
  //                                                    cl_image2d_slice_pitch));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> gen(-5, 5);
  std::vector<float> input_v(batch_size * ic * ih * iw);

  LOG(INFO) << "gen input ...";

  float* input_v_data = &input_v[0];
  for (auto& i : input_v) {
    i = gen(engine);
  }
  paddle::lite::CLImageConverterDefault default_convertor;

  std::vector<float> x_image_data(input_image_width * input_image_height *
                                  4);  // 4 : RGBA

  LOG(INFO) << "set mapped input  ...";
  default_convertor.NCHWToImage(input_v_data, x_image_data.data(), input_dim);

  auto* input_image = input.mutable_data<float, cl::Image2D>(
      input_image_width, input_image_height, x_image_data.data());

  LOG(INFO) << "prepare kernel ready";

  LOG(INFO) << "mutable output ...";
  CLImageConverterDefault default_converter;
  DDim out_image_shape = default_converter.InitImageDimInfoWith(output_dim);
  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  auto* out_image = output.mutable_data<float, cl::Image2D>(out_image_shape[0],
                                                            out_image_shape[1]);
  VLOG(4) << "out_dims= " << output_dim;

  LOG(INFO) << "kernel context ...";
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  std::unique_ptr<KernelContext> reshape_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(reshape_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(reshape_context));

  LOG(INFO) << "kernel launch ...";
  kernel->Launch();

  auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto* out_ptr = param.output->data<float, cl::Image2D>();
  auto it = wait_list->find(out_image);

  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto& event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  float* out_image_data = new float[out_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(out_image_data,
                              output.data<float, cl::Image2D>(),
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  float* out_data = new float[out_image_shape.production() * 4];
  default_converter.ImageToNCHW(
      out_image_data, out_data, out_image_shape, output_dim);
  // check output dims
  for (int i = 0; i < output.dims().size(); i++) {
    CHECK_EQ(output.dims()[i], shape_tensor_data[i]);
  }

  // check output data
  for (int i = 0; i < output.numel(); i++) {
    EXPECT_NEAR(out_data[i], input_v_data[i], 1e-3);
    if (abs(out_data[i] - input_v_data[i]) > 1e-3) {
      LOG(INFO) << "error idx:" << i;
    }
  }
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(reshape, kOpenCL, kFloat, kImageDefault, image2d);
USE_LITE_KERNEL(reshape2, kOpenCL, kFloat, kImageDefault, image2d);
