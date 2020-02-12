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
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/utils/logging.h"

namespace paddle {
namespace lite {

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
                       bool flag_relu) {
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
            if (flag_relu) {
              dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                          ? dst_data_ref[out_idx]
                                          : (Dtype2)0;
            }
          }
        }
      }
    }
  }
}
TEST(conv2d_1x1, compute) {
  // conv infos
  const int ksize = 1;
  const int stride = 1;
  const int pad = 0;
  const int group = 1;
  const int dilation = 0;
//  int loop_cnt = 0;

#ifdef LOOP_TEST
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 4; oc < 10; oc += 1) {                       // oc
      for (int ih = 4; ih < 9; ih += 1) {                      // ih
        /*int iw = ih;*/ for (int iw = 4; iw < 10; iw += 1) {  // iw
          for (int ic = 4; ic < 10; ic += 1) {                 // ic
            for (bool bias_flag : {true, false}) {
              for (bool relu_flag : {true, false}) {
#else
  const int batch_size = 1;
  const int oc = 4;
  const int ih = 8;
  const int iw = 8;
  const int ic = 4;
  const bool bias_flag = false;
  const bool relu_flag = false;
#endif
                const int oh = ih;
                const int ow = iw;

                VLOG(4) << "to get kernel ...";
                auto kernels =
                    KernelRegistry::Global().Create("conv2d_1x1",
                                                    TARGET(kOpenCL),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kImageDefault));
                ASSERT_FALSE(kernels.empty());

                auto kernel = std::move(kernels.front());
                VLOG(4) << "created conv2d_1x1 kernel";

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
                std::uniform_real_distribution<float> gen(-5, 5);

                std::vector<float> input_v(batch_size * ic * ih * iw);
                std::vector<float> filter_v(oc * ic * ksize * ksize);
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
                VLOG(4) << "filter_dim.production(): "
                        << filter_dim.production();
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
                std::vector<float> x_image_v(
                    input_image_width * input_image_height * 4);  // 4 : RGBA
                std::vector<float> filter_image_v(
                    filter_image_width * filter_image_height * 4);  // 4 : RGBA
                std::vector<float> bias_image_v(
                    bias_image_width * bias_image_height * 4);  // 4 : RGBA
                std::vector<float> out_image_v(
                    out_image_width * out_image_height * 4);  // 4 : RGBA

                default_convertor.NCHWToImage(
                    input_v.data(), x_image_v.data(), input_dim);

                /*                for (int j = 0; j < input_v.size(); j += 1) {
                                  //                  VLOG(4) << "input_v
                   input[" << j << "]:
                                  //                  " << input_v.data()[j];
                                  std::cout << j << "  " << input_v.data()[j] <<
                   std::endl;
                                }
                                std::cout << std::endl;

                                for (int j = 0; j < x_image_v.size(); j += 1) {
                                  //                  VLOG(4) << "x_image_v
                   input[" << j <<
                                  //                  "]: " <<
                   x_image_v.data()[j];
                                  std::cout << j << "  " << x_image_v.data()[j]
                   << std::endl;
                                }*/

                VLOG(4) << "set mapped filter  ...";
                paddle::lite::CLImageConverterNWBlock nw_convertor;
                nw_convertor.NCHWToImage(
                    filter_v.data(), filter_image_v.data(), filter_dim);

                auto* input_image2d = input.mutable_data<float, cl::Image2D>(
                    input_image_width, input_image_height, x_image_v.data());
                auto* filter_image2d = filter.mutable_data<float, cl::Image2D>(
                    filter_image_width,
                    filter_image_height,
                    filter_image_v.data());

                if (bias_flag) {
                  nw_convertor.NCHWToImage(
                      filter_v.data(), filter_image_v.data(), filter_dim);

                  for (int i = 0; i < bias_dim.production(); ++i) {
                    bias_v[i] = static_cast<int>(gen(engine));
                  }
                  CLImageConverterFolder folder_convertor;
                  folder_convertor.NCHWToImage(
                      bias_v.data(), bias_image_v.data(), bias_dim);
                  auto* bias_data = bias.mutable_data<float, cl::Image2D>(
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
                auto* output_image2d = output.mutable_data<float, cl::Image2D>(
                    out_image_width, out_image_height);

                auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
                auto* out_ptr = param.output->data<float, cl::Image2D>();
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
                                            output.data<float, cl::Image2D>(),
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
                VLOG(4) << "mutable_data out_ref_data: ";

                // run cpu ref
                auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));

                VLOG(4) << " conv_basic beigin ..... ";

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
                VLOG(4) << " conv_basic end ..... ";

                VLOG(4) << " out_dim: " << out_dim;
                const DDim& out_image_dims = lite::DDim{std::vector<int64_t>(
                    {static_cast<int64_t>(out_image_width),
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
      }
    }
  }
#else
// nothing to do.
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d_1x1, kOpenCL, kFloat, kImageDefault, image2d);
