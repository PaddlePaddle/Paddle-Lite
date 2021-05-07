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
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

#define SHADOW_LOG VLOG(4)
#define FP16_RELATIVE_DIFF (5e-2)
#define FP16_ABS_DIFF (5e-2)
#define FP32_RELATIVE_DIFF (1e-3)
#define FP32_ABS_DIFF (5e-4)
#define TEST_DEPTHWISE_CONV_IMAGE_BASIC
#define TEST_DEPTHWISE_CONV_IMAGE_3X3

#define LEAKY_RELU_ALPHA (0.1)
template <typename Dtype1, typename Dtype2>
static void conv_basic(const Dtype1* din,
                       Dtype2* dout,
                       int num,
                       int chout,
                       int hout,
                       int wout,
                       int chin,
                       int hin,
                       int win,
                       const Dtype1* weights,
                       const Dtype2* bias,
                       int group,
                       int kernel_w,
                       int kernel_h,
                       int stride_w,
                       int stride_h,
                       int dila_w,
                       int dila_h,
                       int pad_w,
                       int pad_h,
                       bool flag_bias,
                       std::string flag_relu,
                       float leaky_relu_alpha = LEAKY_RELU_ALPHA) {
  auto src_data = din;
  auto dst_data_ref = dout;
  auto weights_data = weights;
  auto with_bias = flag_bias;
  auto bias_data = bias;

  int in_num = num;
  int out_channels = chout;
  int out_h = hout;
  int out_w = wout;

  int in_channel = chin;
  int in_h = hin;
  int in_w = win;
  int out_c_group = out_channels / group;
  int in_c_group = in_channel / group;

  for (int n = 0; n < in_num; ++n) {
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * group * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            Dtype2 bias_d =
                with_bias ? (bias_data[g * out_c_group + oc]) : (Dtype2)0;
            dst_data_ref[out_idx] = bias_d;
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int iw = ow * stride_w - pad_w + kw * (dila_w);
                  int ih = oh * stride_h - pad_h + kh * (dila_h);
                  if (iw < 0 || iw >= in_w) continue;
                  if (ih < 0 || ih >= in_h) continue;

                  int iidx = n * in_channel * in_h * in_w +
                             g * in_c_group * in_h * in_w + ic * in_h * in_w +
                             ih * in_w + iw;
                  int widx =
                      g * out_c_group * in_c_group * kernel_h * kernel_w +
                      oc * in_c_group * kernel_h * kernel_w +
                      ic * kernel_h * kernel_w + kh * kernel_w + kw;

                  dst_data_ref[out_idx] += src_data[iidx] * weights_data[widx];
                }
              }
            }

            if (flag_relu == "relu") {
              dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                          ? dst_data_ref[out_idx]
                                          : (Dtype2)0;
            } else if (flag_relu == "relu6") {
              auto dst_tmp = (dst_data_ref[out_idx] > (Dtype2)0)
                                 ? dst_data_ref[out_idx]
                                 : (Dtype2)0;
              dst_data_ref[out_idx] = (dst_tmp < 6.f) ? dst_tmp : 6.f;
            } else if (flag_relu == "leaky_relu") {
              dst_data_ref[out_idx] =
                  dst_data_ref[out_idx] > (Dtype2)0
                      ? dst_data_ref[out_idx]
                      : (Dtype2)(dst_data_ref[out_idx] * leaky_relu_alpha);
            } else if (flag_relu == "") {
              continue;  // no act
            } else {
              LOG(FATAL) << "this act type: " << flag_relu
                         << " does not support";
            }
          }
        }
      }
    }
  }
}

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

#ifdef TEST_DEPTHWISE_CONV_IMAGE_BASIC
// #define LOOP_TEST
void test_precision(const lite_api::CLPrecisionType p) {
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  const int fc = 1;
  const int fw = 7;
  const int fh = fw;
  const int dilation_h = 1;
  const int dilation_w = 2;
  const int stride_h = 2;
  const int stride_w = 3;
  const int pad_h = 0;
  const int pad_w = 3;
  const bool bias_flag = true;
  const std::string act_flag = "leaky_relu";
#ifdef LOOP_TEST
  // for (int batch_size = 1; batch_size < 2; ++batch_size) {
  for (int ic = 4; ic < 10; ic += 1) {
    for (int ih = 3; ih < 15; ih += 1) {
      for (int iw = 3; iw < 15; iw += 1) {
#else
  const int ic = 6;
  const int ih = 112;
  const int iw = 112;
#endif
        const int fb = ic;
        const int oc = ic;
        const int oh =
            ConvOutputSize(ih, fh, dilation_h, pad_h, pad_h, stride_h);
        const int ow =
            ConvOutputSize(iw, fw, dilation_w, pad_w, pad_w, stride_w);
        if (oh <= 0 || ow <= 0) {
#ifdef LOOP_TEST
          continue;
#else
    LOG(FATAL) << "Output tensor of depthwise conv is illegal!"
               << "Please check your input dims and conv params";
#endif
        }

        LOG(INFO) << "to get kernel ...";
        auto kernels =
            KernelRegistry::Global().Create("depthwise_conv2d",
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
        param.groups = oc;
        std::vector<int> paddings = {pad_h, pad_h, pad_w, pad_w};
        param.paddings = std::make_shared<std::vector<int>>(paddings);
        param.strides = std::vector<int>{stride_h, stride_w};
        std::vector<int> dilations = {dilation_h, dilation_w};
        param.dilations = std::make_shared<std::vector<int>>(dilations);
        param.bias = bias_flag ? &bias : nullptr;

        if (act_flag == "relu") {
          param.fuse_relu = true;
          param.activation_param.has_active = true;
          param.activation_param.active_type = lite_api::ActivationType::kRelu;
        } else if (act_flag == "relu6") {
          param.activation_param.Relu_clipped_coef = 6.f;
          param.activation_param.has_active = true;
          param.activation_param.active_type = lite_api::ActivationType::kRelu6;
        } else if (act_flag == "leaky_relu") {
          param.activation_param.active_type =
              lite_api::ActivationType::kLeakyRelu;
          param.activation_param.has_active = true;
          param.activation_param.Leaky_relu_alpha = LEAKY_RELU_ALPHA;
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
        const DDim& input_dim =
            lite::DDim{std::vector<int64_t>({1, ic, ih, iw})};
        const DDim& filter_dim =
            lite::DDim{std::vector<int64_t>({fb, fc, fh, fw})};
        const DDim& output_dim =
            lite::DDim{std::vector<int64_t>({1, oc, oh, ow})};
        // channel wise bias
        const DDim bias_dim = DDim(std::vector<DDim::value_type>{oc});
        input.Resize(input_dim);
        filter.Resize(filter_dim);
        output.Resize(output_dim);

        std::vector<float> input_v(input_dim.production());
        std::vector<float> filter_v(filter_dim.production());
        std::vector<float> output_v(output_dim.production());
        fill_data_rand(input_v.data(), -1.f, 1.f, input_dim.production());
        fill_data_rand(filter_v.data(), -1.f, 1.f, filter_dim.production());
        fill_data_const(output_v.data(), 0.f, output_dim.production());
        std::vector<float> bias_v;

        LOG(INFO) << "prepare input";
        CLImageConverterDefault* default_converter =
            new CLImageConverterDefault();
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

        LOG(INFO) << "prepare kernel";
        filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(),
                                                       filter_dim);
        if (bias_flag) {
          bias.Resize(bias_dim);
          bias_v.resize(bias_dim.production());
          fill_data_rand(bias_v.data(), -1.f, 1.f, bias_dim.production());
          bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(), bias_dim);
        }

        LOG(INFO) << "launch";
        DDim output_image_shape =
            default_converter->InitImageDimInfoWith(output.dims());
        LOG(INFO) << "output_image_shape = " << output_image_shape[0] << " "
                  << output_image_shape[1];
        auto* output_image = MUTABLE_DATA_GPU(
            &output, output_image_shape[0], output_image_shape[1], nullptr);

        kernel->Launch();

        CLRuntime::Global()->command_queue().finish();

        lite::Tensor out_ref;
        out_ref.Resize(output_dim);
        auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

        conv_basic<float, float>(input_v.data(),
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
                                 pad_w,
                                 pad_h,
                                 bias_flag,
                                 act_flag);

        const size_t cl_image2d_row_pitch{0};
        const size_t cl_image2d_slice_pitch{0};

        std::vector<char> output_image_data(output_image_shape.production() *
                                            4 * dtype_size);  // 4 : RGBA
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

        LOG(INFO) << "output_data vs output_ref_data";
        auto relative_diff_thres =
            fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
        auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
        uint32_t diff_cnt = 0;
        for (int i = 0; i < output.dims().production(); i++) {
          auto relative_diff =
              COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
          auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
          EXPECT_FALSE(relative_diff > relative_diff_thres &&
                       abs_diff > abs_diff_thres);
          if (relative_diff > relative_diff_thres &&
              abs_diff > abs_diff_thres) {
            LOG(WARNING) << "err idx: " << i << " abs_diff: " << abs_diff
                         << "\t relative_diff: " << relative_diff
                         << "\t out_ins: " << output_v[i]
                         << "\t out_ref: " << out_ref_data[i];
            diff_cnt++;
          }
        }
        if (diff_cnt != 0) {
          LOG(FATAL) << "Err num " << diff_cnt << "/"
                     << output_dim.production();
        }
#ifdef LOOP_TEST
      }
    }
  }
#else
// nothing to do.
#endif
  LOG(INFO) << "\n\t[  PASSED  ] " << lite_api::CLPrecisionTypeToStr(p);
}

TEST(depthwise_conv2d, compute_basic) {
  for (auto ins : {lite_api::CLPrecisionType::CL_PRECISION_FP32,
                   lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    test_precision(ins);
  }
}
#endif

#ifdef TEST_DEPTHWISE_CONV_IMAGE_3X3
#define LOOP_TEST
TEST(depthwise_conv2d, compute_image2d_3x3) {
  const int fc = 1;
  const int fw = 3;
  const int fh = fw;
  const int dilation = 4;
  const int stride = 2;
  const int pad = 2;
  const bool bias_flag = false;
  const std::string relu_flag = "leaky_relu";
#ifdef LOOP_TEST
  // for (int batch_size = 1; batch_size < 2; ++batch_size) {
  for (int oc = 4; oc < 10; oc += 1) {      // oc = ic
    for (int ih = 3; ih < 15; ih += 1) {    // ih
      for (int iw = 3; iw < 15; iw += 1) {  // iw
#else
  const int oc = 32;
  const int ih = 112;
  const int iw = 112;
#endif
        const int fb = oc;
        const int ic = oc;
        const int oh = ConvOutputSize(ih, fh, dilation, pad, pad, stride);
        const int ow = ConvOutputSize(iw, fw, dilation, pad, pad, stride);
        if (oh <= 0 || ow <= 0) {
#ifdef LOOP_TEST
          continue;
#else
    LOG(FATAL) << "Output tensor of depthwise conv is illegal!"
               << "Please check your input dims and conv params";
#endif
        }

        LOG(INFO) << "to get kernel ...";
        auto kernels =
            KernelRegistry::Global().Create("depthwise_conv2d",
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
        param.groups = oc;
        std::vector<int> paddings = {pad, pad, pad, pad};
        param.paddings = std::make_shared<std::vector<int>>(paddings);
        param.strides = std::vector<int>{stride, stride};
        std::vector<int> dilations = {dilation, dilation};
        param.dilations = std::make_shared<std::vector<int>>(dilations);
        param.bias = bias_flag ? &bias : nullptr;

        if (relu_flag == "relu") {
          param.fuse_relu = true;  // relu only
          param.activation_param.has_active = true;
          param.activation_param.active_type = lite_api::ActivationType::kRelu;
        } else if (relu_flag == "relu6") {
          param.activation_param.Relu_clipped_coef = 6.f;
          param.activation_param.has_active = true;
          param.activation_param.active_type = lite_api::ActivationType::kRelu6;
        } else if (relu_flag == "leaky_relu") {
          param.activation_param.active_type =
              lite_api::ActivationType::kLeakyRelu;
          param.activation_param.has_active = true;
          param.activation_param.Leaky_relu_alpha = LEAKY_RELU_ALPHA;
        } else {
          param.fuse_relu = false;  // relu only
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
        const DDim& input_dim =
            lite::DDim{std::vector<int64_t>({1, ic, ih, iw})};
        const DDim& filter_dim =
            lite::DDim{std::vector<int64_t>({fb, fc, fh, fw})};
        const DDim& output_dim =
            lite::DDim{std::vector<int64_t>({1, oc, oh, ow})};
        // channel wise bias
        const DDim bias_dim = DDim(std::vector<DDim::value_type>{oc});
        input.Resize(input_dim);
        filter.Resize(filter_dim);
        output.Resize(output_dim);

        std::default_random_engine engine;
        std::uniform_real_distribution<float> gen(-5, 5);
        std::vector<float> input_v(input_dim.production());
        std::vector<float> filter_v(filter_dim.production());
        std::vector<float> output_v(output_dim.production());
        for (auto& i : input_v) {
          i = gen(engine);
        }
        for (auto& f : filter_v) {
          f = gen(engine);
        }
        std::vector<float> bias_v;
        if (bias_flag) {
          bias.Resize(bias_dim);
          bias_v.resize(bias_dim.production());
          for (auto& b : bias_v) {
            b = gen(engine);
          }
        }

        LOG(INFO) << "prepare input";
        CLImageConverterDefault* default_converter =
            new CLImageConverterDefault();
        DDim input_image_shape =
            default_converter->InitImageDimInfoWith(input.dims());
        LOG(INFO) << "input_image_shape = " << input_image_shape[0] << " "
                  << input_image_shape[1];
        std::vector<half_t> input_image_data(input_image_shape.production() *
                                             4);  // 4 : RGBA
        default_converter->NCHWToImage(
            input_v.data(), input_image_data.data(), input.dims());
        auto* input_image =
            input.mutable_data<half_t, cl::Image2D>(input_image_shape[0],
                                                    input_image_shape[1],
                                                    input_image_data.data());

        LOG(INFO) << "prepare kernel";
        filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(),
                                                       filter_dim);
        if (bias_flag) {
          bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(), bias_dim);
        }

        LOG(INFO) << "launch";
        DDim output_image_shape =
            default_converter->InitImageDimInfoWith(output.dims());
        LOG(INFO) << "output_image_shape = " << output_image_shape[0] << " "
                  << output_image_shape[1];
        auto* output_image = output.mutable_data<half_t, cl::Image2D>(
            output_image_shape[0], output_image_shape[1]);

        kernel->Launch();

        CLRuntime::Global()->command_queue().finish();

        lite::Tensor out_ref;
        out_ref.Resize(output_dim);
        auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

        conv_basic<float, float>(input_v.data(),
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
                                 stride,
                                 stride,
                                 dilation,
                                 dilation,
                                 pad,
                                 pad,
                                 bias_flag,
                                 relu_flag);

        const size_t cl_image2d_row_pitch{0};
        const size_t cl_image2d_slice_pitch{0};

        std::vector<half_t> output_image_data(output_image_shape.production() *
                                              4);
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

        LOG(INFO) << "output_data vs output_ref_data";
        for (int i = 0; i < output.dims().production(); i++) {
          auto relative_diff =
              COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
          auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
          EXPECT_FALSE(relative_diff > FP16_RELATIVE_DIFF &&
                       abs_diff > FP16_ABS_DIFF);
          if (relative_diff > FP16_RELATIVE_DIFF && abs_diff > FP16_ABS_DIFF) {
            LOG(FATAL) << "error idx:" << i << " output_v[" << i
                       << "]:" << output_v[i] << " "
                                                 "out_ref_data["
                       << i << "]:" << out_ref_data[i];
          }
        }
#ifdef LOOP_TEST
      }
    }
  }
#else
// nothing to do.
#endif
}
#endif

}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(depthwise_conv2d, kOpenCL, kFP16, kImageDefault, image2d);
