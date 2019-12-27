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

#include "lite/kernels/fpga/conv_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

static int get_rand(int start, int end) {
  int i = rand();  // NOLINT
  i = (i % (end - start)) + start;
  return i;
}

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

template <typename Dtype1, typename Dtype2>
void conv_compute_ref(const operators::ConvParam& param) {
  const Dtype1* din = param.x->data<Dtype1>();
  Dtype2* dout = param.output->mutable_data<Dtype2>();

  int num = param.x->dims()[0];
  int chout = param.output->dims()[1];
  int hout = param.output->dims()[2];
  int wout = param.output->dims()[3];

  int chin = param.x->dims()[1];
  int hin = param.x->dims()[2];
  int win = param.x->dims()[3];

  const Dtype1* weights = param.filter->mutable_data<Dtype1>();
  Dtype2* bias = nullptr;
  if (param.bias != nullptr) {
    bias = param.bias->mutable_data<Dtype2>();
  }

  int group = param.groups;
  int kernel_w = param.filter->dims()[2];
  int kernel_h = param.filter->dims()[3];
  int stride_w = param.strides[0];
  int stride_h = param.strides[1];
  int dila_w = (*param.dilations)[0];
  int dila_h = (*param.dilations)[1];

  int pad_w = (*param.paddings)[2];
  int pad_h = (*param.paddings)[0];
  bool flag_bias = (param.bias != nullptr);
  bool flag_relu = param.fuse_relu;

  conv_basic(din,
             dout,
             num,
             chout,
             hout,
             wout,
             chin,
             hin,
             win,
             weights,
             bias,
             group,
             kernel_w,
             kernel_h,
             stride_w,
             stride_h,
             dila_w,
             dila_h,
             pad_w,
             pad_h,
             flag_bias,
             flag_relu);
}

TEST(conv_fpga, retrive_op) {
  auto elementwise_add =
      KernelRegistry::Global()
          .Create<TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)>("conv2d");
  ASSERT_FALSE(elementwise_add.empty());
  ASSERT_TRUE(elementwise_add.front());
}

TEST(conv_fpga, init) {
  ConvCompute conv;
  ASSERT_EQ(conv.precision(), PRECISION(kFP16));
  ASSERT_EQ(conv.target(), TARGET(kFPGA));
}

TEST(conv_fpga, compute) {
  DeviceInfo::Init();
#if 1
  for (auto n : {2}) {
    for (auto ic : {6}) {
      for (auto oc : {6}) {
        for (auto ih : {9}) {
          for (auto iw : {9}) {
            for (auto flag_bias : {false, true}) {
              for (auto flag_relu : {false, true}) {
                for (auto depthwise : {false, true}) {
                  for (auto dilation : {1}) {
                    for (auto stride : {1, 2}) {
                      for (auto padding : {0, 1, 2}) {
                        for (auto ks : {1, 3, 5}) {
#else
  for (auto n : {1, 2}) {
    for (auto ic : {6, 32 /*, 128*/}) {
      for (auto oc : {6, 32 /*, 128*/}) {
        for (auto ih : {9, 18 /*, 56 , 112, 224, 512*/}) {
          for (auto iw : {9, 18 /*, 56, 112, 224, 512*/}) {
            for (auto flag_bias : {false, true}) {
              for (auto flag_relu : {false, true}) {
                for (auto depthwise : {false, true}) {
                  for (auto dilation : {1, 2}) {
                    for (auto stride : {1, 2}) {
                      for (auto padding : {0, 1, 2}) {
                        for (auto ks : {1, 3, 5}) {
#endif
                          int group = 1;
                          if (depthwise) {  // depthwise convolution ?
                            group = oc = ic;
                          }
                          // get input, filter and output shape
                          std::vector<int64_t> input_shape = {n, ic, ih, iw};
                          std::vector<int64_t> filter_shape = {
                              oc, ic / group, ks, ks};
                          const int dks = dilation * (ks - 1) + 1;
                          int oh = (ih + 2 * padding - dks) / stride + 1;
                          int ow = (iw + 2 * padding - dks) / stride + 1;
                          std::vector<int64_t> output_shape({n, oc, oh, ow});
                          // resize input, filter and outfloat16put
                          Tensor input;
                          Tensor filter;
                          Tensor bias;
                          Tensor output;
                          Tensor output_ref;
                          input.Resize(input_shape);
                          filter.Resize(filter_shape);
                          output.Resize(output_shape);
                          output_ref.Resize(output_shape);
                          VLOG(3) << "input: " << input.dims();
                          VLOG(3) << "filter: " << filter.dims()
                                  << " padding:" << padding
                                  << " stride:" << stride
                                  << " dilation:" << dilation;
                          VLOG(3) << "output: " << output.dims();
                          auto* input_data =
                              input.mutable_data<float16>(TARGET(kFPGA));
                          auto* filter_data =
                              filter.mutable_data<float>(TARGET(kFPGA));
                          auto* output_data =
                              output.mutable_data<float16>(TARGET(kFPGA));
                          for (int i = 0; i < input.dims().production(); i++) {
                            float sign = i % 3 == 0 ? -1.0f : 1.0f;
                            input_data[i] = sign * static_cast<float>(i % 128);
                          }
                          for (int i = 0; i < filter.dims().production(); i++) {
                            filter_data[i] =
                                i * 0.001f /
                                static_cast<float>(filter.dims().production());
                          }
                          // prepare kernel params and run
                          ConvCompute conv;
                          operators::ConvParam param;
                          param.x = &input;
                          param.filter = &filter;
                          param.output = &output;
                          param.bias = nullptr;
                          if (flag_bias) {
                            bias.Resize({oc});
                            auto* bias_data = bias.mutable_data<float>();
                            for (int i = 0; i < bias.dims().production(); i++) {
                              bias_data[i] = static_cast<float>(i);
                            }
                            param.bias = &bias;
                          }
                          param.fuse_relu = flag_relu;
                          *param.paddings = std::vector<int>(
                              {padding, padding, padding, padding});
                          param.strides = std::vector<int>({stride, stride});
                          *param.dilations =
                              std::vector<int>({dilation, dilation});
                          param.groups = group;
                          conv.SetParam(param);
                          conv.Launch();
                          // invoking ref implementation and compare results
                          param.output = &output_ref;
                          conv_compute_ref<float, float>(param);
                          auto* output_ref_data =
                              output_ref.mutable_data<float>();
                          for (int i = 0; i < output.dims().production(); i++) {
                            EXPECT_NEAR(
                                output_data[i], output_ref_data[i], 1e-3);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kFPGA, kFP16, kNHWC, def);
