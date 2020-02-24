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

#include <iostream>
#include <random>

#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

template <typename T, int STRIDE_H = 1, int STRIDE_W = 1>
void depth_conv(const T* input_data,
                const lite::DDim& input_dims,
                const T* filter_data,
                const lite::DDim& filter_dims,
                T* output_data,
                const lite::DDim& output_dims) {
  int stride_h = STRIDE_H, stride_w = STRIDE_W;

  int64_t batches = input_dims[0];
  int64_t channels = input_dims[1];
  int64_t h = input_dims[2];
  int64_t w = input_dims[3];

  int64_t num_output = output_dims[1];
  int64_t outh = output_dims[2];
  int64_t outw = output_dims[3];

  int64_t filter_h = filter_dims[2];
  int64_t filter_w = filter_dims[3];

  const int64_t in_batch_size = channels * h * w;
  const int64_t out_batch_size = num_output * outh * outw;

  auto kernel_offset = std::unique_ptr<int[]>(new int[filter_h * filter_w]);
  {
    int p = 0;
    int offset = 0;
    int gap = w - filter_w;
    for (int i = 0; i < filter_h; i++) {
      for (int j = 0; j < filter_w; j++) {
        kernel_offset[p++] = offset;
        offset += 1;
      }
      offset += gap;
    }
  }

  for (int b = 0; b < batches; b++) {
    auto* input_batch_start = input_data + b * in_batch_size;
    auto* output_batch_start = output_data + b * out_batch_size;
    for (int p = 0; p < num_output; p++) {
      float* output_ptr = output_batch_start + p * outh * outw;
      const float* filter_ptr = filter_data + p * filter_h * filter_w;
      const float* input_ptr = input_batch_start + p * h * w;

      for (int i = 0; i < outh; i++) {
        for (int j = 0; j < outw; j++) {
          float sum = 0;
          const float* input_ch_start =
              input_ptr + i * stride_h * w + j * stride_w;

          for (int fh = 0; fh < filter_h; ++fh) {
            for (int fw = 0; fw < filter_w; ++fw) {
              float val = input_ch_start[kernel_offset[fh * filter_w + fw]];
              float w = filter_ptr[fh * filter_w + fw];
              sum += val * w;
            }
          }
          output_ptr[j] = sum;
        }

        output_ptr += outw;
      }
    }
  }
}
int ConvOutputSize(int input_size,
                   int filter_size,
                   int dilation,
                   int pad_left,
                   int pad_right,
                   int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
}

// #define LOOP_TEST
TEST(depthwise_conv2d_basic, compute) {
  // conv infos
  //  const int ksize = 1;
  const int stride = 1;
  const int pad = 0;
  const int group = 1;
  const int dilation = 1;
  const int fc = 1;
  const int batch_size = 1;
  const int bias_flag = false;
  const bool relu_flag = false;

//  int loop_cnt = 0;

#ifdef LOOP_TEST
  // for (int batch_size = 1; batch_size < 2; ++batch_size) {
  for (int oc = 4; oc < 10; oc += 1) {         // oc = ic
    for (int fw = 3; fw < 10; fw += 2) {       // fh = fw
      for (int ih = fw; ih < 15; ih += 1) {    // ih
        for (int iw = fw; iw < 15; iw += 1) {  // iw
#else
  const int oc = 32;
  const int ih = 112;
  const int iw = 112;
  const int fw = 5;

#endif

          const int fb = oc;
          const int ic = oc;
          const int fh = fw;

          const int oh = ConvOutputSize(ih, fh, dilation, pad, pad, stride);
          const int ow = ConvOutputSize(iw, fw, dilation, pad, pad, stride);

          VLOG(4) << "to get kernel ...";
          auto kernels =
              KernelRegistry::Global().Create("depthwise_conv2d",
                                              TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault));
          ASSERT_FALSE(kernels.empty());

          auto kernel = std::move(kernels.front());
          VLOG(4) << "created depthconv2d kernel";

          VLOG(4) << "prepare kernel ------";

          lite::Tensor input, filter, bias, output;
          operators::ConvParam param;
          param.x = &input;
          param.filter = &filter;
          param.output = &output;
          if (bias_flag) {
            param.bias = &bias;
          }
          param.fuse_relu = relu_flag;

          std::vector<int> paddings = {pad, pad, pad, pad};
          std::vector<int> dilations = {dilation, dilation};

          param.paddings = std::make_shared<std::vector<int>>(paddings);
          param.dilations = std::make_shared<std::vector<int>>(dilations);
          param.strides = std::vector<int>{stride, stride};

          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          std::unique_ptr<KernelContext> depth_conv_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(depth_conv_context->As<OpenCLContext>()));
          kernel->SetContext(std::move(depth_conv_context));

          const DDim& input_dim =
              lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};

          const DDim& filter_dim =
              lite::DDim{std::vector<int64_t>({fb, fc, fh, fw})};
          const DDim& out_dim =
              lite::DDim{std::vector<int64_t>({batch_size, oc, oh, ow})};
          // element wise bias
          const DDim& bias_dim = lite::DDim{std::vector<int64_t>({oc})};

          param.x->Resize(input_dim);
          param.filter->Resize(filter_dim);
          param.output->Resize(out_dim);
          if (bias_flag) {
            param.bias->Resize(bias_dim);
          }

          kernel->SetParam(param);

          size_t input_image_width = iw * ((ic + 3) / 4);
          size_t input_image_height = ih * batch_size;

          size_t out_image_width = ow * ((oc + 3) / 4);
          size_t out_image_height = oh * batch_size;

          size_t bias_image_width = ow * ((oc + 3) / 4);
          size_t bias_image_height = oh * batch_size;

          size_t filter_image_width = fw * ((fb + 3) / 4);
          size_t filter_image_height = fc * fh;

          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};

          std::default_random_engine engine;
          std::uniform_real_distribution<float> gen(-5, 5);

          std::vector<float> input_v(batch_size * ic * ih * iw);
          std::vector<float> filter_v(fb * fc * fh * fw);
          std::vector<float> output_v(batch_size * oc * ih * iw);
          std::vector<float> bias_v(oc);

          VLOG(4) << "gen input and filter ...";

          for (auto& i : input_v) {
            i = gen(engine);
          }
          for (auto& f : filter_v) {
            f = gen(engine);
          }

          VLOG(4) << "after gen input and filter ...";
          VLOG(4) << "input_v.size(): " << input_v.size();
          VLOG(4) << "filter_v.size(): " << filter_v.size();
          VLOG(4) << "output_v.size(): " << output_v.size();
          VLOG(4) << "bias_v.size(): " << bias_v.size();
          VLOG(4) << "input_dim.production(): " << input_dim.production();
          VLOG(4) << "filter_dim.production(): " << filter_dim.production();
          VLOG(4) << "out_dim.production(): " << out_dim.production();
          VLOG(4) << "bias_dim.production(): " << bias_dim.production();
          VLOG(4) << "4 * input_image_height * input_image_width: "
                  << 4 * input_image_height * input_image_width;
          VLOG(4) << "4 * filter_image_width * filter_image_height: "
                  << 4 * filter_image_width * filter_image_height;

          CHECK(input_dim.production() == input_v.size());
          CHECK_LE(input_dim.production(),
                   4 * input_image_height * input_image_width);
          CHECK(filter_dim.production() == filter_v.size());
          CHECK_LE(filter_dim.production(),
                   4 * filter_image_width * filter_image_height);

          paddle::lite::CLImageConverterDefault default_convertor;
          VLOG(4) << "set mapped input  ...";
          std::vector<uint16_t> x_image_v(input_image_width *
                                          input_image_height * 4);  // 4 : RGBA
          std::vector<uint16_t> filter_image_v(
              filter_image_width * filter_image_height * 4);  // 4 : RGBA
          std::vector<uint16_t> bias_image_v(
              bias_image_width * bias_image_height * 4);  // 4 : RGBA
          std::vector<uint16_t> out_image_v(out_image_width * out_image_height *
                                            4);  // 4 : RGBA

          default_convertor.NCHWToImage(
              input_v.data(), x_image_v.data(), input_dim);

          VLOG(4) << "set mapped filter  ...";
          paddle::lite::CLImageConverterNWBlock nw_convertor;
          nw_convertor.NCHWToImage(
              filter_v.data(), filter_image_v.data(), filter_dim);

          auto* input_image2d = input.mutable_data<uint16_t, cl::Image2D>(
              input_image_width, input_image_height, x_image_v.data());
          auto* filter_image2d = filter.mutable_data<uint16_t, cl::Image2D>(
              filter_image_width, filter_image_height, filter_image_v.data());

          if (bias_flag) {
            nw_convertor.NCHWToImage(
                filter_v.data(), filter_image_v.data(), filter_dim);

            for (int i = 0; i < bias_dim.production(); ++i) {
              bias_v[i] = static_cast<int>(gen(engine));
            }
            CLImageConverterFolder folder_convertor;
            folder_convertor.NCHWToImage(
                bias_v.data(), bias_image_v.data(), bias_dim);
            auto* bias_data = bias.mutable_data<uint16_t, cl::Image2D>(
                bias_image_width, bias_image_height, bias_image_v.data());
          }

          VLOG(4) << "resize output  ...";
          output.Resize(out_dim);

          // cpu conv basic calc
          lite::Tensor out_ref;
          out_ref.Resize(out_dim);

          VLOG(4) << "prepare kernel ready";

          VLOG(4) << "kernel launch ...";
          kernel->Launch();
          VLOG(4) << "mutable output ...";
          auto* output_image2d = output.mutable_data<uint16_t, cl::Image2D>(
              out_image_width, out_image_height);

          auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
          auto* out_ptr = param.output->data<uint16_t, cl::Image2D>();
          auto it = wait_list->find(out_ptr);

          if (it != wait_list->end()) {
            VLOG(4) << "--- Find the sync event for the target cl "
                       "tensor. ---";
            auto& event = *(it->second);
            event.wait();
          } else {
            LOG(FATAL) << "Could not find the sync event for the target "
                          "cl tensor.";
          }

          TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                      output.data<uint16_t, cl::Image2D>(),
                                      out_image_width,
                                      out_image_height,
                                      cl_image2d_row_pitch,
                                      cl_image2d_slice_pitch,
                                      IoDirection::DtoH);
          DDim out_image_shape =
              default_convertor.InitImageDimInfoWith(output.dims());

          default_convertor.ImageToNCHW(out_image_v.data(),
                                        output_v.data(),
                                        out_image_shape,
                                        output.dims());

          // for (int j = 0; j < input_v.size(); j += 1) {
          //   VLOG(4) << "input_v input[" << j
          //           << "]: " << input_v.data()[j];
          //       std::cout<< j << "  " << input_v.data()[j] << std::endl;
          // }
          // std::cout << std::endl;

          // for (int j = 0; j < output_v.size(); j += 1) {
          //   VLOG(4) << "output_v output_v[" << j
          //           << "]:" << output_v.data()[j];
          //       std::cout << j << "  " << output_v.data()[j] <<
          //       std::endl;
          // }

          VLOG(4) << "mutable_data out_ref_data: ";

          // run cpu ref
          auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

          VLOG(4) << " conv_basic beigin ..... ";
          depth_conv<float, 1, 1>(input_v.data(),
                                  input.dims(),
                                  filter_v.data(),
                                  filter.dims(),
                                  out_ref_data,
                                  out_dim);
          VLOG(4) << " conv_basic end ..... ";

          VLOG(4) << " input_dim: " << input_dim;
          VLOG(4) << " filter_dim: " << filter_dim;
          const DDim& out_image_dims = lite::DDim{
              std::vector<int64_t>({static_cast<int64_t>(out_image_width),
                                    static_cast<int64_t>(out_image_height)})};

          for (int i = 0; i < out_dim.production(); i++) {
            EXPECT_NEAR(output_v[i], out_ref_data[i], 1e-2);
            if (abs(output_v[i] - out_ref_data[i]) > 1e-2) {
              LOG(FATAL) << "error idx:" << i;
            }
          }

#ifdef LOOP_TEST
        }
      }
    }
  }
#else
// nothing to do.
#endif
}

TEST(depthwise_conv2d_image2d_fp16, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create("depthwise_conv2d",
                                                 TARGET(kOpenCL),
                                                 PRECISION(kFP16),
                                                 DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";
  lite::Tensor input, filter, output;
  operators::ConvParam param;
  param.x = &input;
  param.filter = &filter;
  param.output = &output;
  std::vector<int> paddings = {0, 0};
  param.paddings = std::make_shared<std::vector<int>>(paddings);
  param.strides = std::vector<int>{1, 1};
  std::vector<int> dilations = {1, 1};
  param.dilations = std::make_shared<std::vector<int>>(dilations);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(dep_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(dep_context));

  LOG(INFO) << "kernel ready";
  std::default_random_engine engine;
  std::uniform_real_distribution<float> gen(-5, 5);
  std::vector<float> input_v(1 * 32 * 112 * 112);
  std::vector<float> filter_v(32 * 1 * 3 * 3);
  for (auto& i : input_v) {
    i = gen(engine);
  }
  for (auto& f : filter_v) {
    f = gen(engine);
  }

  LOG(INFO) << "prepare input";
  input.Resize({1, 32, 112, 112});
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim input_image_shape =
      default_converter->InitImageDimInfoWith(input.dims());
  LOG(INFO) << "input_image_shape = " << input_image_shape[0] << " "
            << input_image_shape[1];
  std::vector<uint16_t> input_image_data(input_image_shape.production() *
                                         4);  // 4 : RGBA
  default_converter->NCHWToImage(
      input_v.data(), input_image_data.data(), input.dims());
  auto* input_image = input.mutable_data<uint16_t, cl::Image2D>(
      input_image_shape[0], input_image_shape[1], input_image_data.data());

  LOG(INFO) << "prepare kernel";
  filter.Resize({32, 1, 3, 3});
  CLImageConverterNWBlock* nw_converter = new CLImageConverterNWBlock();
  DDim filter_image_shape = nw_converter->InitImageDimInfoWith(filter.dims());
  LOG(INFO) << "filter_image_shape = " << filter_image_shape[0] << " "
            << filter_image_shape[1];
  std::vector<uint16_t> filter_image_data(filter_image_shape.production() *
                                          4);  // 4 : RGBA
  nw_converter->NCHWToImage(
      filter_v.data(), filter_image_data.data(), filter.dims());
  auto* filter_image = filter.mutable_data<uint16_t, cl::Image2D>(
      filter_image_shape[0], filter_image_shape[1], filter_image_data.data());

  LOG(INFO) << "launch";
  output.Resize({1, 32, 110, 110});
  DDim output_image_shape =
      default_converter->InitImageDimInfoWith(output.dims());
  LOG(INFO) << "output_image_shape = " << output_image_shape[0] << " "
            << output_image_shape[1];
  auto* output_image = output.mutable_data<uint16_t, cl::Image2D>(
      output_image_shape[0], output_image_shape[1]);

  kernel->Launch();

  auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto* out_ptr = param.output->data<uint16_t, cl::Image2D>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    LOG(INFO) << "--- Find the sync event for the target cl tensor. ---";
    auto& event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
    LOG(INFO) << "Could not find the sync event for the target cl tensor.";
  }

  lite::Tensor output_ref;
  output_ref.Resize({1, 32, 110, 110});
  auto* output_ref_data = output_ref.mutable_data<float>(TARGET(kARM));
  depth_conv<float, 1, 1>(input_v.data(),
                          input.dims(),
                          filter_v.data(),
                          filter.dims(),
                          output_ref_data,
                          output_ref.dims());

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};

  uint16_t* output_image_data =
      new uint16_t[output_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(output_image_data,
                              output_image,
                              output_image_shape[0],
                              output_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);

  float* output_data = new float[output_image_shape.production() * 4];
  default_converter->ImageToNCHW(
      output_image_data, output_data, output_image_shape, output.dims());

  LOG(INFO) << "output_data vs output_ref_data";
  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-4);
    LOG(INFO) << output_data[i] << " " << output_ref_data[i];
  }
}

}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(depthwise_conv2d, kOpenCL, kFP16, kImageDefault, image2d);
