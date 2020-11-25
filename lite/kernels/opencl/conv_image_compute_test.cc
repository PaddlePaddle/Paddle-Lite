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

#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

namespace paddle {
namespace lite {
// #define SHADOW_LOG LOG(INFO)
#define SHADOW_LOG VLOG(4)
#define FP16_MAX_DIFF (1e0)
#define FP16_ABS_DIFF (1e-1)
// #define TEST_CONV_IMAGE_ALL_1
#define TEST_CONV_IMAGE_1x1
#define TEST_CONV_IMAGE_3x3
#define TEST_CONV_IMAGE_5x5
#define TEST_CONV_IMAGE_7x7

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
  Dtype2 beta = 0;
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
            dst_data_ref[out_idx] = bias_d;  // + dst_data_ref[out_idx] * beta;
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
            } else {
              VLOG(4) << "this act type: " << flag_relu << " does not support";
            }
          }
        }
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

#ifdef TEST_CONV_IMAGE_1x1
// #define PRINT_RESULT
// #define LOOP_TEST
TEST(conv2d, compute_image2d_1x1) {
  // conv infos
  const int ksize = 1;
  const int stride = 1;
  const int pad = 0;
  const int group = 1;
  const int dilation = 0;
//  int loop_cnt = 0;

#ifdef LOOP_TEST
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 2; oc < 10; oc += 1) {   // oc
      for (int ih = 2; ih < 9; ih += 1) {  // ih
        int iw = ih;
        for (int ic = 2; ic < 10; ic += 1) {  // ic
          for (bool bias_flag : {true, false}) {
            for (std::string relu_flag : {"leaky_relu"}) {
#else
  const int batch_size = 1;
  const int oc = 2;
  const int ih = 3;
  const int iw = 3;
  const int ic = 2;
  const bool bias_flag = false;
  const std::string relu_flag = "leaky_relu";
#endif
              LOG(INFO) << "---------------------------- "
                           "conv1x1----------------------- "
                           "run---------------";
              const int oh = ih;
              const int ow = iw;
              LOG(INFO) << "batch_size:  " << batch_size;
              LOG(INFO) << "ic:  " << ic;
              LOG(INFO) << "ih:  " << ih;
              LOG(INFO) << "iw:  " << iw;
              LOG(INFO) << "oc:  " << oc;
              LOG(INFO) << "bias_flag:  " << bias_flag;
              LOG(INFO) << "relu_flag:  " << relu_flag;

              SHADOW_LOG << "to get kernel ...";
              auto kernels =
                  KernelRegistry::Global().Create("conv2d",
                                                  TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault));
              ASSERT_FALSE(kernels.empty());

              auto kernel = std::move(kernels.front());
              SHADOW_LOG << "created conv2d_1x1 kernel";

              SHADOW_LOG << "prepare kernel ------";

              lite::Tensor input, filter, bias, output;
              operators::ConvParam param;
              param.x = &input;
              param.filter = &filter;
              param.output = &output;
              if (bias_flag) {
                param.bias = &bias;
              }

              if (relu_flag == "relu") {
                param.fuse_relu = true;  // relu only
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu;
              } else if (relu_flag == "relu6") {
                param.activation_param.Relu_clipped_coef = 6.f;
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu6;
              } else if (relu_flag == "leaky_relu") {
                param.activation_param.active_type =
                    lite_api::ActivationType::kLeakyRelu;
                param.activation_param.has_active = true;
                param.activation_param.Leaky_relu_alpha = LEAKY_RELU_ALPHA;
              } else {
                param.fuse_relu = false;  // relu only
                param.activation_param.has_active = false;
              }

              std::vector<int> paddings = {pad, pad, pad, pad};
              std::vector<int> dilations = {dilation, dilation};

              param.paddings = std::make_shared<std::vector<int>>(paddings);
              param.dilations = std::make_shared<std::vector<int>>(dilations);
              param.strides = std::vector<int>{stride, stride};

              std::unique_ptr<KernelContext> context(new KernelContext);
              context->As<OpenCLContext>().InitOnce();

              std::unique_ptr<KernelContext> conv_1x1_context(
                  new KernelContext);
              context->As<OpenCLContext>().CopySharedTo(
                  &(conv_1x1_context->As<OpenCLContext>()));
              kernel->SetContext(std::move(conv_1x1_context));

              const DDim& input_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};

              const DDim& filter_dim =
                  lite::DDim{std::vector<int64_t>({oc, ic, ksize, ksize})};
              const DDim& out_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, oc, ih, iw})};
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

              size_t filter_image_width = ksize * ((oc + 3) / 4);
              size_t filter_image_height = ic * ksize;

              const size_t cl_image2d_row_pitch{0};
              const size_t cl_image2d_slice_pitch{0};

              std::default_random_engine engine;
              std::uniform_real_distribution<float> gen(-2, 2);

              std::vector<float> input_v(batch_size * ic * ih * iw);
              std::vector<float> filter_v(oc * ic * ksize * ksize);
              std::vector<float> output_v(batch_size * oc * ih * iw);
              std::vector<float> bias_v(oc);

              SHADOW_LOG << "gen input and filter ...";

              for (auto& i : input_v) {
                i = gen(engine);
#ifdef TEST_CONV_IMAGE_ALL_1
                i = 0.01;
#endif
              }
              for (auto& f : filter_v) {
                f = gen(engine);
#ifdef TEST_CONV_IMAGE_ALL_1
                f = 0.01;
#endif
              }

              SHADOW_LOG << "after gen input and filter ...";
              SHADOW_LOG << "input_v.size(): " << input_v.size();
              SHADOW_LOG << "filter_v.size(): " << filter_v.size();
              SHADOW_LOG << "output_v.size(): " << output_v.size();
              SHADOW_LOG << "bias_v.size(): " << bias_v.size();
              SHADOW_LOG << "input_dim.production(): "
                         << input_dim.production();
              SHADOW_LOG << "filter_dim.production(): "
                         << filter_dim.production();
              SHADOW_LOG << "out_dim.production(): " << out_dim.production();
              SHADOW_LOG << "bias_dim.production(): " << bias_dim.production();
              SHADOW_LOG << "4 * input_image_height * input_image_width: "
                         << 4 * input_image_height * input_image_width;
              SHADOW_LOG << "4 * filter_image_width * filter_image_height: "
                         << 4 * filter_image_width * filter_image_height;

              CHECK(input_dim.production() == input_v.size());
              CHECK_LE(input_dim.production(),
                       4 * input_image_height * input_image_width);
              CHECK(filter_dim.production() == filter_v.size());
              CHECK_LE(filter_dim.production(),
                       4 * filter_image_width * filter_image_height);

              paddle::lite::CLImageConverterDefault default_convertor;
              SHADOW_LOG << "set mapped input  ...";
              std::vector<half_t> x_image_v(
                  input_image_width * input_image_height * 4);  // 4 : RGBA
              std::vector<half_t> filter_image_v(
                  filter_image_width * filter_image_height * 4);  // 4 :RGBA
              std::vector<half_t> bias_image_v(
                  bias_image_width * bias_image_height * 4);  // 4 : RGBA
              std::vector<half_t> out_image_v(
                  out_image_width * out_image_height * 4);  // 4 : RGBA

              default_convertor.NCHWToImage(
                  input_v.data(), x_image_v.data(), input_dim);

              SHADOW_LOG << "set mapped filter  ...";
              paddle::lite::CLImageConverterNWBlock nw_convertor;
              nw_convertor.NCHWToImage(
                  filter_v.data(), filter_image_v.data(), filter_dim);

              auto* input_image2d = input.mutable_data<half_t, cl::Image2D>(
                  input_image_width, input_image_height, x_image_v.data());
              // assign filter as target arm
              filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(),
                                                             filter_dim);
              SHADOW_LOG << " lite输入 input_v ..... ";
              for (int i = 0; i < input_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << input_v[i];
              }
              SHADOW_LOG << " lite输入 input_image2d ..... ";
              for (int i = 0; i < x_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << Half2Float(x_image_v[i]);
              }
              SHADOW_LOG << "卷积核 : ----  ";
              for (int i = 0; i < filter_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << filter_v[i];
              }

              SHADOW_LOG << "卷积核1: ----  ";
              const float* filter_p = filter.data<float>();
              for (int i = 0; i < filter_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << *filter_p;
                filter_p++;
              }
              SHADOW_LOG << "卷积核2:  ----  ";
              const float* filter_p2 = filter.mutable_data<float>();
              for (int i = 0; i < filter_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << *filter_p2;
                filter_p2++;
              }

              SHADOW_LOG << "卷积核 image : ----  ";
              for (int i = 0; i < filter_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << Half2Float(filter_image_v[i]);
              }
              if (bias_flag) {
                for (int i = 0; i < bias_dim.production(); ++i) {
                  bias_v[i] = static_cast<int>(gen(engine));
                }
                bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(),
                                                             bias_dim);
              }

              SHADOW_LOG << "resize output  ...";
              output.Resize(out_dim);

              // cpu conv basic calc
              lite::Tensor out_ref;
              out_ref.Resize(out_dim);

              SHADOW_LOG << "prepare kernel ready";

              SHADOW_LOG << "kernel launch ...";
              kernel->Launch();
              SHADOW_LOG << "mutable output ...";
              auto* output_image2d = output.mutable_data<half_t, cl::Image2D>(
                  out_image_width, out_image_height);

              CLRuntime::Global()->command_queue().finish();

              TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                          output.data<half_t, cl::Image2D>(),
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
              SHADOW_LOG << " lite输出 out_image_v ..... ";
              for (int i = 0; i < out_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << Half2Float(out_image_v[i]);
              }
              SHADOW_LOG << " lite输出 output_v ..... ";
              for (int i = 0; i < output_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << output_v[i];
              }
              SHADOW_LOG << "mutable_data out_ref_data: ";

              // run cpu ref
              auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

              SHADOW_LOG << " conv_basic beigin ..... ";

              conv_basic<float, float>(input_v.data(),
                                       out_ref_data,
                                       batch_size,
                                       oc,
                                       oh,
                                       ow,
                                       ic,
                                       ih,
                                       iw,
                                       filter_v.data(),
                                       bias_v.data(),  // mapped_bias,
                                       group,
                                       ksize,
                                       ksize,
                                       stride,
                                       stride,
                                       dilation,
                                       dilation,
                                       pad,
                                       pad,
                                       bias_flag,
                                       relu_flag);
              SHADOW_LOG << " conv_basic end ..... ";

              SHADOW_LOG << " out_dim: " << out_dim;
              const DDim& out_image_dims = lite::DDim{std::vector<int64_t>(
                  {static_cast<int64_t>(out_image_width),
                   static_cast<int64_t>(out_image_height)})};

              for (int i = 0; i < out_dim.production(); i++) {
                auto relative_diff =
                    COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
                auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
                // EXPECT_LT(relative_diff, FP16_MAX_DIFF);
                EXPECT_FALSE(relative_diff > FP16_MAX_DIFF &&
                             abs_diff > FP16_ABS_DIFF);
                if (relative_diff > FP16_MAX_DIFF && abs_diff > FP16_ABS_DIFF) {
                  LOG(FATAL) << "error idx:" << i << "output_v[" << i
                             << "]:" << output_v[i] << " "
                                                       "out_ref_data["
                             << i << "]:" << out_ref_data[i];
                }
              }
#ifdef LOOP_TEST
            }
          }
        }
      }
    }
  }
#else
// nothing to do.
#endif
}
#undef LOOP_TEST
#undef PRINT_RESULT
#endif

#ifdef TEST_CONV_IMAGE_3x3
// #define PRINT_RESULT
// #define LOOP_TEST
TEST(conv2d, compute_image2d_3x3) {
  // conv infos
  const int ksize = 3;
//  int loop_cnt = 0;

#ifdef LOOP_TEST
  const int pad = 1;
  const int dilation = 1;
  const int stride = 2;
  const int group = 1;
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 1; oc < 10; oc += 1) {   // oc
      for (int ih = 5; ih < 9; ih += 1) {  // ih
        int iw = ih;
        for (int ic = 1; ic < 10; ic += 1) {  // ic
          for (bool bias_flag : {true, false}) {
            for (std::string relu_flag : {"", "relu"}) {
#else
  const int pad = 1;
  const int dilation = 1;

#if 0  // small scale with group, but result of cpu reference is wrong
const int stride = 2;
                const int group = 2;
                const int batch_size = 1;
                const int ic = 1;
                const int ih = 3;
                const int iw = 3;
                const int oc = 2;
#else  // big scale with group
  const int stride = 2;
  const int group = 1;
  const int batch_size = 1;
  const int ic = 3 / 1;
  const int ih = 224 / 1;
  const int iw = 112 / 1;
  const int oc = 32 / 1;
#endif

  const bool bias_flag = false;
  const std::string relu_flag = "leaky_relu";
#endif
              int filter_channel = ic;
              if (group > 1) {
                filter_channel = 1;
              }

              const int oh =
                  ConvOutputSize(ih, ksize, dilation, pad, pad, stride);
              const int ow =
                  ConvOutputSize(iw, ksize, dilation, pad, pad, stride);
              SHADOW_LOG << "to get kernel ...";
              auto kernels =
                  KernelRegistry::Global().Create("conv2d",
                                                  TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault));
              ASSERT_FALSE(kernels.empty());
              auto kernel = std::move(kernels.front());
              SHADOW_LOG << "created conv2d kernel";

              SHADOW_LOG << "prepare kernel ------";

              lite::Tensor input, filter, bias, output;
              operators::ConvParam param;
              param.x = &input;
              param.filter = &filter;
              param.output = &output;
              param.groups = group;
              if (bias_flag) {
                param.bias = &bias;
              }

              if (relu_flag == "relu") {
                param.fuse_relu = true;  // relu only
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu;
              } else if (relu_flag == "relu6") {
                param.activation_param.Relu_clipped_coef = 6.f;
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu6;
              } else if (relu_flag == "leaky_relu") {
                param.activation_param.active_type =
                    lite_api::ActivationType::kLeakyRelu;
                param.activation_param.has_active = true;
                param.activation_param.Leaky_relu_alpha = LEAKY_RELU_ALPHA;
              } else {
                param.fuse_relu = false;  // relu only
                param.activation_param.has_active = false;
              }

              std::vector<int> paddings = {pad, pad, pad, pad};
              std::vector<int> dilations = {dilation, dilation};

              param.paddings = std::make_shared<std::vector<int>>(paddings);
              param.dilations = std::make_shared<std::vector<int>>(dilations);
              param.strides = std::vector<int>{stride, stride};

              std::unique_ptr<KernelContext> context(new KernelContext);
              context->As<OpenCLContext>().InitOnce();

              std::unique_ptr<KernelContext> conv_1x1_context(
                  new KernelContext);
              context->As<OpenCLContext>().CopySharedTo(
                  &(conv_1x1_context->As<OpenCLContext>()));
              kernel->SetContext(std::move(conv_1x1_context));

              const DDim& input_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};

              const DDim& filter_dim = lite::DDim{
                  std::vector<int64_t>({oc, filter_channel, ksize, ksize})};
              const DDim& out_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, oc, oh, ow})};
              // element wise bias
              const DDim& bias_dim = lite::DDim{std::vector<int64_t>({oc})};

              VLOG(2) << "input_dim:" << input_dim
                      << " filter_dim:" << filter_dim << " out_dim:" << out_dim
                      << " bias_flag:" << bias_flag << " bias_dim:" << bias_dim
                      << " group:" << group << " stride:" << stride
                      << " pad:" << pad << " dilation:" << dilation;

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

              size_t filter_image_width = ksize * ((filter_channel + 3) / 4);
              size_t filter_image_height = oc * ksize;

              const size_t cl_image2d_row_pitch{0};
              const size_t cl_image2d_slice_pitch{0};

              std::default_random_engine engine;
              std::uniform_real_distribution<float> gen(-2, 2);

              std::vector<float> input_v(batch_size * ic * ih * iw);
              std::vector<float> filter_v(oc * filter_channel * ksize * ksize);
              std::vector<float> output_v(batch_size * oc * oh * ow);
              std::vector<float> bias_v(oc);

              SHADOW_LOG << "gen input and filter ...";
              for (int i = 0; i < input_v.size(); ++i) {
                input_v[i] = gen(engine);
              }
              for (int i = 0; i < filter_v.size(); ++i) {
                filter_v[i] = gen(engine);
              }

              SHADOW_LOG << "after gen input and filter ...";
              SHADOW_LOG << "input_v.size(): " << input_v.size();
              SHADOW_LOG << "filter_v.size(): " << filter_v.size();
              SHADOW_LOG << "output_v.size(): " << output_v.size();
              SHADOW_LOG << "bias_v.size(): " << bias_v.size();
              SHADOW_LOG << "input_dim.production(): "
                         << input_dim.production();
              SHADOW_LOG << "filter_dim.production(): "
                         << filter_dim.production();
              SHADOW_LOG << "out_dim.production(): " << out_dim.production();
              SHADOW_LOG << "bias_dim.production(): " << bias_dim.production();
              SHADOW_LOG << "input_image_height:" << input_image_height
                         << " input_image_width:" << input_image_width;
              SHADOW_LOG << "filter_image_height:" << filter_image_height
                         << " filter_image_width:" << filter_image_width;
              SHADOW_LOG << "4 * input_image_height *input_image_width: "
                         << 4 * input_image_height * input_image_width;
              SHADOW_LOG << "4 * filter_image_width * filter_image_height: "
                         << 4 * filter_image_width * filter_image_height;

              CHECK(input_dim.production() == input_v.size());
              CHECK_LE(input_dim.production(),
                       4 * input_image_height * input_image_width);
              CHECK(filter_dim.production() == filter_v.size());
              CHECK_LE(filter_dim.production(),
                       4 * filter_image_width * filter_image_height);

              paddle::lite::CLImageConverterDefault default_convertor;
              SHADOW_LOG << "set mapped input  ...";
              std::vector<half_t> x_image_v(input_image_width *
                                            input_image_height * 4);  // 4 :RGBA
              std::vector<half_t> filter_image_v(
                  filter_image_width * filter_image_height * 4);  // 4 : RGBA
              std::vector<half_t> bias_image_v(
                  bias_image_width * bias_image_height * 4);  // 4 : RGBA
              std::vector<half_t> out_image_v(out_image_width *
                                              out_image_height * 4);  // 4 :RGBA

              default_convertor.NCHWToImage(
                  input_v.data(), x_image_v.data(), input_dim);
              SHADOW_LOG << "输入: ----  ";
              for (int i = 0; i < input_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << input_v[i];
              }
              SHADOW_LOG << "输入image : ----  ";
              for (int i = 0; i < x_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << x_image_v[i];
              }
              SHADOW_LOG << "set mapped filter  ...";
              CLImageConverterFolder folder_convertor;

              folder_convertor.NCHWToImage(
                  filter_v.data(), filter_image_v.data(), filter_dim);
              SHADOW_LOG << "卷积核: ----  ";
              for (int i = 0; i < filter_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << filter_v[i];
              }
              SHADOW_LOG << "卷积核image: ----  ";
              for (int i = 0; i < filter_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << filter_image_v[i];
              }
              auto* input_image2d = input.mutable_data<half_t, cl::Image2D>(
                  input_image_width, input_image_height, x_image_v.data());
              // assign filter as target arm
              filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(),
                                                             filter_dim);
              if (bias_flag) {
                for (int i = 0; i < bias_dim.production(); ++i) {
                  bias_v[i] = static_cast<int>(gen(engine));
                }
                bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(),
                                                             bias_dim);
              }

              SHADOW_LOG << "resize output  ...";
              output.Resize(out_dim);

              // cpu conv basic calc
              lite::Tensor out_ref;
              out_ref.Resize(out_dim);

              SHADOW_LOG << "prepare kernel ready";

              SHADOW_LOG << "kernel launch ...";
              kernel->Launch();
              SHADOW_LOG << "mutable output ...";
              auto* output_image2d = output.mutable_data<half_t, cl::Image2D>(
                  out_image_width, out_image_height);

              CLRuntime::Global()->command_queue().finish();
              TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                          output.data<half_t, cl::Image2D>(),
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

              SHADOW_LOG << "输出: ----  ";
              for (int i = 0; i < output_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << output_v[i];
              }

              SHADOW_LOG << "输出image: ----  ";
              for (int i = 0; i < out_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << out_image_v[i];
              }
              SHADOW_LOG << "mutable_data out_ref_data: ";

              // run cpu ref
              auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

              SHADOW_LOG << " conv_basic beigin ..... ";

              conv_basic<float, float>(input_v.data(),
                                       out_ref_data,
                                       batch_size,
                                       oc,
                                       oh,
                                       ow,
                                       ic,
                                       ih,
                                       iw,
                                       filter_v.data(),
                                       bias_v.data(),  // mapped_bias,
                                       group,
                                       ksize,
                                       ksize,
                                       stride,
                                       stride,
                                       dilation,
                                       dilation,
                                       pad,
                                       pad,
                                       bias_flag,
                                       relu_flag);
              SHADOW_LOG << " conv_basic end ..... ";

              SHADOW_LOG << " out_dim: " << out_dim;
              const DDim& out_image_dims = lite::DDim{std::vector<int64_t>(
                  {static_cast<int64_t>(out_image_width),
                   static_cast<int64_t>(out_image_height)})};

#ifdef PRINT_RESULT
              for (int i = 0; i < out_dim.production(); i++) {
                VLOG(4) << "output_v[" << i << "]:" << output_v[i]
                        << " out_ref_data[" << i << "]:" << out_ref_data[i];
              }
#endif

              for (int i = 0; i < out_dim.production(); i++) {
                auto relative_diff =
                    COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
                auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
                // EXPECT_LT(relative_diff, FP16_MAX_DIFF);
                // EXPECT_LT(abs_diff, FP16_ABS_DIFF);

                EXPECT_FALSE(relative_diff > FP16_MAX_DIFF &&
                             abs_diff > FP16_ABS_DIFF);
                if (relative_diff > FP16_MAX_DIFF && abs_diff > FP16_ABS_DIFF) {
                  LOG(FATAL) << "error idx:" << i << "output_v[" << i
                             << "]:" << output_v[i] << " "
                                                       "out_ref_data["
                             << i << "]:" << out_ref_data[i];
                }
              }

#ifdef LOOP_TEST
            }
          }
        }
      }
    }
  }
#else
// nothing to do.
#endif
}
#undef LOOP_TEST
#undef PRINT_RESULT

#endif

#ifdef TEST_CONV_IMAGE_5x5

// #define PRINT_RESULT
// #define LOOP_TEST
TEST(conv2d, compute_image2d_5x5) {
  // conv infos
  const int ksize = 5;
  const int stride = 1;
  const int pad = 2;
  const int group = 1;
  const int dilation = 1;
//  int loop_cnt = 0;

#ifdef LOOP_TEST
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 1; oc < 5; oc += 1) {    // oc
      for (int ih = 5; ih < 8; ih += 1) {  // ih
        int iw = ih;
        for (int ic = 2; ic < 6; ic += 1) {  // ic
          for (bool bias_flag : {true, false}) {
            for (std::string relu_flag : {""
                                          "relu"}) {
#else
  const int batch_size = 2;
  const int oc = 1;
  const int ih = 5;
  const int iw = 5;
  // ic = 1 会进入depthwise的路由 .
  const int ic = 2;
  const bool bias_flag = true;
  const std::string relu_flag = "leaky_relu";
#endif

              const int oh =
                  ConvOutputSize(ih, ksize, dilation, pad, pad, stride);
              const int ow =
                  ConvOutputSize(iw, ksize, dilation, pad, pad, stride);
              SHADOW_LOG << "to get kernel ...";
              auto kernels =
                  KernelRegistry::Global().Create("conv2d",
                                                  TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault));
              ASSERT_FALSE(kernels.empty());

              auto kernel = std::move(kernels.front());
              SHADOW_LOG << "created conv2d kernel";

              SHADOW_LOG << "prepare kernel ------";

              lite::Tensor input, filter, bias, output;
              operators::ConvParam param;
              param.x = &input;
              param.filter = &filter;
              param.output = &output;
              if (bias_flag) {
                param.bias = &bias;
              }
              if (relu_flag == "relu") {
                param.fuse_relu = true;  // relu only
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu;
              } else if (relu_flag == "relu6") {
                param.activation_param.Relu_clipped_coef = 6.f;
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu6;
              } else if (relu_flag == "leaky_relu") {
                param.activation_param.active_type =
                    lite_api::ActivationType::kLeakyRelu;
                param.activation_param.has_active = true;
                param.activation_param.Leaky_relu_alpha = LEAKY_RELU_ALPHA;
              } else {
                param.fuse_relu = false;  // relu only
                param.activation_param.has_active = false;
              }

              std::vector<int> paddings = {pad, pad, pad, pad};
              std::vector<int> dilations = {dilation, dilation};

              param.paddings = std::make_shared<std::vector<int>>(paddings);
              param.dilations = std::make_shared<std::vector<int>>(dilations);
              param.strides = std::vector<int>{stride, stride};

              std::unique_ptr<KernelContext> context(new KernelContext);
              context->As<OpenCLContext>().InitOnce();

              std::unique_ptr<KernelContext> conv_1x1_context(
                  new KernelContext);
              context->As<OpenCLContext>().CopySharedTo(
                  &(conv_1x1_context->As<OpenCLContext>()));
              kernel->SetContext(std::move(conv_1x1_context));

              const DDim& input_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};

              const DDim& filter_dim =
                  lite::DDim{std::vector<int64_t>({oc, ic, ksize, ksize})};
              const DDim& out_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, oc, oh, ow})};
              // element wise bias
              const DDim& bias_dim = lite::DDim{std::vector<int64_t>({oc})};

              VLOG(2) << "input_dim:" << input_dim
                      << " filter_dim:" << filter_dim << " out_dim:" << out_dim
                      << " bias_flag:" << bias_flag << " bias_dim:" << bias_dim
                      << " group:" << group << " stride:" << stride
                      << " pad:" << pad << " dilation:" << dilation;

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

              size_t filter_image_width = ksize * ((ic + 3) / 4);
              size_t filter_image_height = oc * ksize;

              const size_t cl_image2d_row_pitch{0};
              const size_t cl_image2d_slice_pitch{0};

              std::default_random_engine engine;
              std::uniform_real_distribution<float> gen(-2, 2);

              std::vector<float> input_v(batch_size * ic * ih * iw);
              std::vector<float> filter_v(oc * ic * ksize * ksize);
              std::vector<float> output_v(batch_size * oc * oh * ow);
              std::vector<float> bias_v(oc);

              SHADOW_LOG << "gen input and filter ...";
              for (auto& i : input_v) {
                i = 0.5 * gen(engine);
              }
              for (auto& f : filter_v) {
                f = 0.5 * gen(engine);
              }

              SHADOW_LOG << "after gen input and filter ...";
              SHADOW_LOG << "input_v.size(): " << input_v.size();
              SHADOW_LOG << "filter_v.size(): " << filter_v.size();
              SHADOW_LOG << "output_v.size(): " << output_v.size();
              SHADOW_LOG << "bias_v.size(): " << bias_v.size();
              SHADOW_LOG << "input_dim.production(): "
                         << input_dim.production();
              SHADOW_LOG << "filter_dim.production(): "
                         << filter_dim.production();
              SHADOW_LOG << "out_dim.production(): " << out_dim.production();
              SHADOW_LOG << "bias_dim.production(): " << bias_dim.production();
              SHADOW_LOG << "4 * input_image_height *input_image_width: "
                         << 4 * input_image_height * input_image_width;
              SHADOW_LOG << "4 * filter_image_width * filter_image_height: "
                         << 4 * filter_image_width * filter_image_height;

              CHECK(input_dim.production() == input_v.size());
              CHECK_LE(input_dim.production(),
                       4 * input_image_height * input_image_width);
              CHECK(filter_dim.production() == filter_v.size());
              CHECK_LE(filter_dim.production(),
                       4 * filter_image_width * filter_image_height);

              paddle::lite::CLImageConverterDefault default_convertor;
              SHADOW_LOG << "set mapped input  ...";
              std::vector<half_t> x_image_v(input_image_width *
                                            input_image_height * 4);  // 4 :RGBA
              std::vector<half_t> filter_image_v(
                  filter_image_width * filter_image_height * 4);  // 4 : RGBA
              std::vector<half_t> bias_image_v(
                  bias_image_width * bias_image_height * 4);  // 4 : RGBA
              std::vector<half_t> out_image_v(out_image_width *
                                              out_image_height * 4);  // 4 :RGBA

              default_convertor.NCHWToImage(
                  input_v.data(), x_image_v.data(), input_dim);
              SHADOW_LOG << "输入: ----  ";
              for (int i = 0; i < input_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << input_v[i];
              }
              SHADOW_LOG << "输入image : ----  ";
              for (int i = 0; i < x_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << x_image_v[i];
              }
              SHADOW_LOG << "set mapped filter  ...";
              CLImageConverterFolder folder_convertor;

              folder_convertor.NCHWToImage(
                  filter_v.data(), filter_image_v.data(), filter_dim);
              SHADOW_LOG << "卷积核: ----  ";
              for (int i = 0; i < filter_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << filter_v[i];
              }
              SHADOW_LOG << "卷积核image: ----  ";
              for (int i = 0; i < filter_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << filter_image_v[i];
              }
              auto* input_image2d = input.mutable_data<half_t, cl::Image2D>(
                  input_image_width, input_image_height, x_image_v.data());
              // assign filter as target arm
              filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(),
                                                             filter_dim);
              if (bias_flag) {
                for (int i = 0; i < bias_dim.production(); ++i) {
                  bias_v[i] = static_cast<int>(gen(engine));
                }
                bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(),
                                                             bias_dim);
              }

              SHADOW_LOG << "resize output  ...";
              output.Resize(out_dim);

              // cpu conv basic calc
              lite::Tensor out_ref;
              out_ref.Resize(out_dim);

              SHADOW_LOG << "prepare kernel ready";

              SHADOW_LOG << "kernel launch ...";
              kernel->Launch();
              SHADOW_LOG << "mutable output ...";
              auto* output_image2d = output.mutable_data<half_t, cl::Image2D>(
                  out_image_width, out_image_height);

              CLRuntime::Global()->command_queue().finish();

              TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                          output.data<half_t, cl::Image2D>(),
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

              SHADOW_LOG << "输出: ----  ";
              for (int i = 0; i < output_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << output_v[i];
              }

              SHADOW_LOG << "输出image: ----  ";
              for (int i = 0; i < out_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << out_image_v[i];
              }
              SHADOW_LOG << "mutable_data out_ref_data: ";

              // run cpu ref
              auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

              SHADOW_LOG << " conv_basic beigin ..... ";

              conv_basic<float, float>(input_v.data(),
                                       out_ref_data,
                                       batch_size,
                                       oc,
                                       oh,
                                       ow,
                                       ic,
                                       ih,
                                       iw,
                                       filter_v.data(),
                                       bias_v.data(),  // mapped_bias,
                                       group,
                                       ksize,
                                       ksize,
                                       stride,
                                       stride,
                                       dilation,
                                       dilation,
                                       pad,
                                       pad,
                                       bias_flag,
                                       relu_flag);
              SHADOW_LOG << " conv_basic end ..... ";

              SHADOW_LOG << " out_dim: " << out_dim;
              const DDim& out_image_dims = lite::DDim{std::vector<int64_t>(
                  {static_cast<int64_t>(out_image_width),
                   static_cast<int64_t>(out_image_height)})};

              for (int i = 0; i < out_dim.production(); i++) {
                auto relative_diff =
                    COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
                auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
                EXPECT_FALSE(relative_diff > FP16_MAX_DIFF &&
                             abs_diff > FP16_ABS_DIFF);
                if (relative_diff > FP16_MAX_DIFF && abs_diff > FP16_ABS_DIFF) {
                  LOG(FATAL) << "error idx:" << i << "output_v[" << i
                             << "]:" << output_v[i] << " "
                                                       "out_ref_data["
                             << i << "]:" << out_ref_data[i];
                }
              }

#ifdef LOOP_TEST
            }
          }
        }
      }
    }
  }
#else
// nothing to do.
#endif
}
#undef LOOP_TEST
#undef PRINT_RESULT
#endif

#ifdef TEST_CONV_IMAGE_7x7
// #undef FP16_ABS_DIFF
// #define FP16_ABS_DIFF (1e-1)
// #define LOOP_TEST
TEST(conv2d, compute_image2d_7x7) {
  // conv infos
  const int ksize = 7;
  const int stride = 1;
  const int pad = 3;
  const int group = 1;
  const int dilation = 1;
//  int loop_cnt = 0;

#ifdef LOOP_TEST
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 1; oc < 6; oc += 1) {    // oc
      for (int ih = 7; ih < 8; ih += 1) {  // ih
        int iw = ih;
        for (int ic = 2; ic < 4; ic += 1) {  // ic
          for (bool bias_flag : {false, true}) {
            for (std::string relu_flag : {"", "relu"}) {
#else
  const int batch_size = 2;
  const int oc = 1;
  const int ih = 7;
  const int iw = 7;
  // ic = 1会进入 depthwise路由
  const int ic = 2;
  const bool bias_flag = false;
  const std::string relu_flag = "leaky_relu";
#endif

              const int oh =
                  ConvOutputSize(ih, ksize, dilation, pad, pad, stride);
              const int ow =
                  ConvOutputSize(iw, ksize, dilation, pad, pad, stride);
              SHADOW_LOG << "to get kernel ...";
              auto kernels =
                  KernelRegistry::Global().Create("conv2d",
                                                  TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault));
              ASSERT_FALSE(kernels.empty());

              auto kernel = std::move(kernels.front());
              SHADOW_LOG << "created conv2d kernel";

              SHADOW_LOG << "prepare kernel ------";

              lite::Tensor input, filter, bias, output;
              operators::ConvParam param;
              param.x = &input;
              param.filter = &filter;
              param.output = &output;
              if (bias_flag) {
                param.bias = &bias;
              }

              if (relu_flag == "relu") {
                param.fuse_relu = true;  // relu only
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu;
              } else if (relu_flag == "relu6") {
                param.activation_param.Relu_clipped_coef = 6.f;
                param.activation_param.has_active = true;
                param.activation_param.active_type =
                    lite_api::ActivationType::kRelu6;
              } else if (relu_flag == "leaky_relu") {
                param.activation_param.active_type =
                    lite_api::ActivationType::kLeakyRelu;
                param.activation_param.has_active = true;
                param.activation_param.Leaky_relu_alpha = LEAKY_RELU_ALPHA;
              } else {
                param.fuse_relu = false;  // relu only
                param.activation_param.has_active = false;
              }

              std::vector<int> paddings = {pad, pad, pad, pad};
              std::vector<int> dilations = {dilation, dilation};

              param.paddings = std::make_shared<std::vector<int>>(paddings);
              param.dilations = std::make_shared<std::vector<int>>(dilations);
              param.strides = std::vector<int>{stride, stride};

              std::unique_ptr<KernelContext> context(new KernelContext);
              context->As<OpenCLContext>().InitOnce();

              std::unique_ptr<KernelContext> conv_1x1_context(
                  new KernelContext);
              context->As<OpenCLContext>().CopySharedTo(
                  &(conv_1x1_context->As<OpenCLContext>()));
              kernel->SetContext(std::move(conv_1x1_context));

              const DDim& input_dim =
                  lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};

              const DDim& filter_dim =
                  lite::DDim{std::vector<int64_t>({oc, ic, ksize, ksize})};
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

              size_t filter_image_width = ksize * ((ic + 3) / 4);
              size_t filter_image_height = oc * ksize;

              const size_t cl_image2d_row_pitch{0};
              const size_t cl_image2d_slice_pitch{0};

              std::default_random_engine engine;
              std::uniform_real_distribution<float> gen(-2, 2);

              std::vector<float> input_v(batch_size * ic * ih * iw);
              std::vector<float> filter_v(oc * ic * ksize * ksize);
              std::vector<float> output_v(batch_size * oc * oh * ow);
              std::vector<float> bias_v(oc);

              SHADOW_LOG << "gen input and filter ...";
              for (auto& i : input_v) {
                i = 0.1 * gen(engine);
#ifdef TEST_CONV_IMAGE_ALL_1
                i = 1;
#endif
              }
              int fiii = 1;
              for (auto& f : filter_v) {
                f = 0.1 * gen(engine);
#ifdef TEST_CONV_IMAGE_ALL_1
                // f = fiii++;
                f = 1;
#endif
              }
              LOG(INFO) << "bias: " << bias_flag;
              LOG(INFO) << "relu: " << relu_flag;

              LOG(INFO) << "inputdims : " << input_dim;
              LOG(INFO) << "filterdims: " << filter.dims();
              LOG(INFO) << "outputdims : " << output.dims();
              SHADOW_LOG << "after gen input and filter ...";
              SHADOW_LOG << "input_v.size(): " << input_v.size();
              SHADOW_LOG << "filter_v.size(): " << filter_v.size();
              SHADOW_LOG << "output_v.size(): " << output_v.size();
              SHADOW_LOG << "bias_v.size(): " << bias_v.size();
              SHADOW_LOG << "input_dim.production(): "
                         << input_dim.production();
              SHADOW_LOG << "filter_dim.production(): "
                         << filter_dim.production();
              SHADOW_LOG << "out_dim.production(): " << out_dim.production();
              SHADOW_LOG << "bias_dim.production(): " << bias_dim.production();
              SHADOW_LOG << "4 * input_image_height * input_image_width: "
                         << 4 * input_image_height * input_image_width;
              SHADOW_LOG << "4 * filter_image_width * filter_image_height: "
                         << 4 * filter_image_width * filter_image_height;

              CHECK(input_dim.production() == input_v.size());
              CHECK_LE(input_dim.production(),
                       4 * input_image_height * input_image_width);
              CHECK(filter_dim.production() == filter_v.size());
              CHECK_LE(filter_dim.production(),
                       4 * filter_image_width * filter_image_height);

              paddle::lite::CLImageConverterDefault default_convertor;
              SHADOW_LOG << "set mapped input  ...";
              std::vector<half_t> x_image_v(
                  input_image_width * input_image_height * 4);  // 4 : RGBA
              std::vector<half_t> filter_image_v(
                  filter_image_width * filter_image_height * 4);  // 4 : RGBA
              std::vector<half_t> bias_image_v(
                  bias_image_width * bias_image_height * 4);  // 4 : RGBA
              std::vector<half_t> out_image_v(
                  out_image_width * out_image_height * 4);  // 4 : RGBA

              default_convertor.NCHWToImage(
                  input_v.data(), x_image_v.data(), input_dim);
              SHADOW_LOG << "输入: ----  ";
              for (int i = 0; i < input_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << input_v[i];
              }
              SHADOW_LOG << "输入image : ----  ";
              for (int i = 0; i < x_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << Half2Float(x_image_v[i]);
              }
              SHADOW_LOG << "set mapped filter  ...";
              CLImageConverterFolder folder_convertor;

              folder_convertor.NCHWToImage(
                  filter_v.data(), filter_image_v.data(), filter_dim);
              SHADOW_LOG << "卷积核: ----  ";
              for (int i = 0; i < filter_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << filter_v[i];
              }
              SHADOW_LOG << "卷积核image: ----  ";
              for (int i = 0; i < filter_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << Half2Float(filter_image_v[i]);
              }
              auto* input_image2d = input.mutable_data<half_t, cl::Image2D>(
                  input_image_width, input_image_height, x_image_v.data());

              // assign filter as target arm
              filter.Assign<float, lite::DDim, TARGET(kARM)>(filter_v.data(),
                                                             filter_dim);
              if (bias_flag) {
                for (int i = 0; i < bias_dim.production(); ++i) {
                  bias_v[i] = static_cast<int>(gen(engine));
                }
                bias.Assign<float, lite::DDim, TARGET(kARM)>(bias_v.data(),
                                                             bias_dim);
              }

              SHADOW_LOG << "resize output  ...";
              output.Resize(out_dim);

              // cpu conv basic calc
              lite::Tensor out_ref;
              out_ref.Resize(out_dim);

              SHADOW_LOG << "prepare kernel ready";

              SHADOW_LOG << "kernel launch ...";
              kernel->Launch();
              SHADOW_LOG << "mutable output ...";
              auto* output_image2d = output.mutable_data<half_t, cl::Image2D>(
                  out_image_width, out_image_height);

              CLRuntime::Global()->command_queue().finish();

              TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                          output.data<half_t, cl::Image2D>(),
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

              SHADOW_LOG << "输出: ----  ";
              for (int i = 0; i < output_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << output_v[i];
              }

              SHADOW_LOG << "输出image: ----  ";
              for (int i = 0; i < out_image_v.size(); i++) {
                SHADOW_LOG << "(" << i << ")" << Half2Float(out_image_v[i]);
              }
              SHADOW_LOG << "mutable_data out_ref_data: ";

              // run cpu ref
              auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

              SHADOW_LOG << " conv_basic beigin ..... ";

              conv_basic<float, float>(input_v.data(),
                                       out_ref_data,
                                       batch_size,
                                       oc,
                                       oh,
                                       ow,
                                       ic,
                                       ih,
                                       iw,
                                       filter_v.data(),
                                       bias_v.data(),  // mapped_bias,
                                       group,
                                       ksize,
                                       ksize,
                                       stride,
                                       stride,
                                       dilation,
                                       dilation,
                                       pad,
                                       pad,
                                       bias_flag,
                                       relu_flag);
              SHADOW_LOG << " conv_basic end ..... ";

              SHADOW_LOG << " out_dim: " << out_dim;
              const DDim& out_image_dims = lite::DDim{std::vector<int64_t>(
                  {static_cast<int64_t>(out_image_width),
                   static_cast<int64_t>(out_image_height)})};

              for (int i = 0; i < out_dim.production(); i++) {
                auto relative_diff =
                    COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
                auto abs_diff = COMPUTE_ABS_DIFF(output_v[i], out_ref_data[i]);
                EXPECT_FALSE(relative_diff > FP16_MAX_DIFF &&
                             abs_diff > FP16_ABS_DIFF);
                if (relative_diff > FP16_MAX_DIFF && abs_diff > FP16_ABS_DIFF) {
                  LOG(FATAL) << "error idx:" << i << "output_v[" << i
                             << "]:" << output_v[i] << " "
                                                       "out_ref_data["
                             << i << "]:" << out_ref_data[i];
                }
              }

#ifdef LOOP_TEST
            }
          }
        }
      }
    }
  }
#else
// nothing to do.
#endif
}
#undef LOOP_TEST
#undef PRINT_RESULT
#endif

#undef SHADOW_LOG
#undef TEST_CONV_IMAGE_1x1
#undef TEST_CONV_IMAGE_3x3
#undef TEST_CONV_IMAGE_5x5
#undef TEST_CONV_IMAGE_7x7
#undef TEST_CONV_IMAGE_ALL_1

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kOpenCL, kFP16, kImageDefault, image2d);
