// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <tuple>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
namespace paddle {
namespace lite {

#define FP16_RELATIVE_DIFF (5e-2)
#define FP16_ABS_DIFF (5e-2)
#define FP32_RELATIVE_DIFF (1e-3)
#define FP32_ABS_DIFF (5e-4)
/* ic ih iw stride_h stride_w pad_up pad_down pad_left pad_right fh fw bias_flag
 * relu_flag */
std::vector<std::tuple<int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       bool,
                       bool>>
    initialize = {{4, 4, 4, 1, 3, 1, 0, 1, 0, 2, 2, false, false},
                  {8, 8, 8, 2, 2, 1, 0, 1, 0, 2, 2, false, false},
                  {12, 8, 8, 1, 3, 1, 0, 1, 0, 4, 4, false, false},
                  {20, 48, 48, 2, 2, 1, 0, 1, 0, 3, 3, true, true},
                  {24, 60, 60, 1, 3, 1, 0, 1, 0, 4, 4, true, true}};

DDim ConvTransposeOutputSize(const DDim& dim_in,
                             const paddle::lite::operators::ConvParam& param) {
  auto filter_dims = param.filter->dims();
  DDim output_shape = dim_in;
  output_shape[1] = filter_dims[1] * param.groups;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  // out=(in-1)×stride+k-2p
  for (int i = 0; i < 2; i++) {
    int kernel_extent = dilations[i] * (filter_dims[i + 2] - 1) + 1;
    int output_len = (dim_in[i + 2] - 1) * param.strides[i] + kernel_extent -
                     (paddings[2 * i] + paddings[2 * i + 1]);
    output_shape[i + 2] = output_len;
  }
  return output_shape;
}

#define LOOP_TEST
void test_precision(const lite_api::CLPrecisionType p) {
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  for (int i = 0; i < initialize.size(); ++i) {
    const int ic = std::get<0>(initialize[i]);
    const int ih = std::get<1>(initialize[i]);
    const int iw = std::get<2>(initialize[i]);
    const int fb = ic;
    const int fc = 1;
    const int fw = std::get<10>(initialize[i]);
    const int fh = fw;
    const int dilation_h = 1;  // only support dilation == 1
    const int dilation_w = 1;  // only support dilation == 1
    const int stride_h = std::get<3>(initialize[i]);
    const int stride_w = std::get<4>(initialize[i]);
    const int pad_up = std::get<5>(initialize[i]);
    const int pad_down = std::get<6>(initialize[i]);
    const int pad_left = std::get<7>(initialize[i]);
    const int pad_right = std::get<8>(initialize[i]);
    const bool bias_flag = std::get<11>(initialize[i]);
    const bool relu_flag = std::get<12>(initialize[i]);

    LOG(INFO) << "to get kernel ...";
    auto kernels = KernelRegistry::Global().Create("depthwise_conv2d_transpose",
                                                   TARGET(kOpenCL),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kImageDefault));
    ASSERT_FALSE(kernels.empty());
    auto kernel = std::move(kernels.front());

    LOG(INFO) << "get kernel";
    lite::Tensor input, filter, bias, output;
    operators::ConvParam param;
    param.x = &input;
    param.filter = &filter;
    param.output = &output;
    param.groups = ic;
    std::vector<int> paddings = {pad_up, pad_down, pad_left, pad_right};
    param.paddings = std::make_shared<std::vector<int>>(paddings);
    param.strides = std::vector<int>{stride_h, stride_w};
    std::vector<int> dilations = {dilation_h, dilation_w};
    param.dilations = std::make_shared<std::vector<int>>(dilations);
    param.bias = bias_flag ? &bias : nullptr;

    if (relu_flag) {
      param.fuse_relu = true;
      param.activation_param.has_active = true;
      param.activation_param.active_type = lite_api::ActivationType::kRelu;
    } else {
      param.fuse_relu = false;
      param.activation_param.has_active = false;
    }

    std::unique_ptr<KernelContext> context(new KernelContext);
    context->As<OpenCLContext>().InitOnce();

    kernel->SetParam(param);
    std::unique_ptr<KernelContext> dep_context(new KernelContext);
    context->As<OpenCLContext>().CopySharedTo(
        &(dep_context->As<OpenCLContext>()));
    kernel->SetContext(std::move(dep_context));

    LOG(INFO) << "kernel ready";
    const DDim& input_dim = lite::DDim{std::vector<int64_t>({1, ic, ih, iw})};
    const DDim& filter_dim = lite::DDim{std::vector<int64_t>({fb, fc, fh, fw})};

    input.Resize(input_dim);
    filter.Resize(filter_dim);

    const DDim& output_dim = ConvTransposeOutputSize(input_dim, param);
    const int oc = output_dim[1];
    const int oh = output_dim[2];
    const int ow = output_dim[3];
    std::cout << "output_dim: ";
    for (int i = 0; i < output_dim.size(); ++i) {
      std::cout << output_dim[i] << " ";
    }
    std::cout << std::endl;
    const DDim bias_dim = DDim(std::vector<DDim::value_type>{oc});
    output.Resize(output_dim);
    std::vector<float> input_v(input_dim.production());
    std::vector<float> filter_v(filter_dim.production());
    std::vector<float> output_v(output_dim.production());
    std::vector<float> bias_v;
    fill_data_rand(input_v.data(), -1.f, 1.f, input_dim.production());
    fill_data_rand(filter_v.data(), -1.f, 1.f, filter_dim.production());

    LOG(INFO) << "prepare image data for opencl";
    CLImageConverterDefault* default_converter = new CLImageConverterDefault();
    DDim input_image_shape =
        default_converter->InitImageDimInfoWith(input.dims());
    LOG(INFO) << "input_image_shape = " << input_image_shape[0] << " "
              << input_image_shape[1];
    const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
    std::vector<char> input_image_data(input_image_shape.production() * 4 *
                                       dtype_size);  // 4 : RGBA
    default_converter->NCHWToImage(
        input_v.data(), input_image_data.data(), input.dims());
    MUTABLE_DATA_GPU(&input,
                     input_image_shape[0],
                     input_image_shape[1],
                     input_image_data.data());
    filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(), filter_dim);
    if (bias_flag) {
      bias.Resize(bias_dim);
      bias_v.resize(bias_dim.production());
      fill_data_rand(bias_v.data(), -1.f, 1.f, bias_dim.production());
      bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(), bias_dim);
    }
    DDim output_image_shape =
        default_converter->InitImageDimInfoWith(output.dims());
    LOG(INFO) << "output_image_shape = " << output_image_shape[0] << " "
              << output_image_shape[1];
    auto* output_image = MUTABLE_DATA_GPU(
        &output, output_image_shape[0], output_image_shape[1], nullptr);

    LOG(INFO) << "kernel launch";
    kernel->Launch();

    CLRuntime::Global()->command_queue().finish();

    const size_t cl_image2d_row_pitch{0};
    const size_t cl_image2d_slice_pitch{0};
    std::vector<char> output_image_data(output_image_shape.production() * 4 *
                                        dtype_size);  // 4 : RGBA

    TargetWrapperCL::ImgcpySync(output_image_data.data(),
                                output_image,
                                output_image_shape[0],
                                output_image_shape[1],
                                cl_image2d_row_pitch,
                                cl_image2d_slice_pitch,
                                IoDirection::DtoH);
    default_converter->ImageToNCHW(output_image_data.data(),
                                   output_v.data(),
                                   output_image_shape,
                                   output.dims());

    std::vector<float> out_ref(output_dim.production());
    auto* out_ref_data = out_ref.data();
    deconv_basic<float, float>(input_v.data(),
                               out_ref_data,
                               1,
                               oc,
                               oh,
                               ow,
                               ic,
                               ih,
                               iw,
                               filter_v.data(),
                               bias_v.data(),
                               param.groups,
                               fw,
                               fh,
                               stride_w,
                               stride_h,
                               dilation_w,
                               dilation_h,
                               paddings[2],
                               paddings[3],
                               paddings[0],
                               paddings[1],
                               bias_flag,
                               relu_flag);

    LOG(INFO) << "output_data vs output_ref_data";
    auto relative_diff_thres =
        fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
    auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
    uint32_t diff_cnt = 0;
    for (int i = 0; i < output.dims().production(); i++) {
      auto relative_diff = COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
      auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
      EXPECT_FALSE(relative_diff > relative_diff_thres &&
                   abs_diff > abs_diff_thres);
      if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
        LOG(WARNING) << "err idx: " << i << "\t out_ins: " << output_v[i]
                     << "\t out_ref: " << out_ref_data[i];
        diff_cnt++;
      }
    }
    if (diff_cnt != 0) {
      LOG(FATAL) << "Err num " << diff_cnt << "/" << output_dim.production();
    }
  }
  LOG(INFO) << "\n[  PASSED  ] " << lite_api::CLPrecisionTypeToStr(p);
}

TEST(depthwise_conv2d_transpose, compute_basic) {
  for (auto ins : {lite_api::CLPrecisionType::CL_PRECISION_FP32,
                   lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    test_precision(ins);
  }
}

}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(
    depthwise_conv2d_transpose, kOpenCL, kFP16, kImageDefault, image2d);
