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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/tensor_utils.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/conv_compute.h"
#endif  // LITE_WITH_ARM
typedef __fp16 float16_t;
DEFINE_int32(power_mode,
             3,
             "power mode: "
             "0 for POWER_HIGH;"
             "1 for POWER_LOW;"
             "2 for POWER_FULL;"
             "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_bool(basic_test, true, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 112, "input height");
DEFINE_int32(in_width, 112, "input width");

DEFINE_int32(out_channel, 32, "output channel");
DEFINE_int32(group, 1, "group");
DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h0, 1, "pad top");
DEFINE_int32(pad_h1, 1, "pad bottom");
DEFINE_int32(pad_w0, 1, "pad left");
DEFINE_int32(pad_w1, 1, "pad right");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_int32(flag_act,
             0,
             "do activation");  // 0-no act, 1-relu, 2-relu6, 4-leakyrelu
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_bool(flag_bias, true, "with bias");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ConvParam ConvParam;
typedef paddle::lite::operators::ActivationParam ActivationParam;

using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::ConvParam& param) {
  DDim dim_out = dim_in;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  dim_out[1] = param.filter->dims()[0];
  auto kernel_h = param.filter->dims()[2];
  auto kernel_w = param.filter->dims()[3];
  auto h = dim_in[2];
  auto w = dim_in[3];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_top = paddings[0];
  int pad_bottom = paddings[1];
  int pad_left = paddings[2];
  int pad_right = paddings[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  auto hout = (h + pad_top + pad_bottom - kernel_exten) / stride_h + 1;
  kernel_exten = dila_w * (kernel_w - 1) + 1;
  auto wout = (w + pad_left + pad_right - kernel_exten) / stride_w + 1;
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

#ifdef LITE_WITH_ARM
void test_conv_fp16(const std::vector<DDim>& input_dims,
                    const DDim& weight_dim,
                    int group,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    int flag_act,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode,
                    const float leakey_relu_scale) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  ConvParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFP16));
  param.filter = new Tensor;
  param.filter->Resize(weight_dim);
  param.filter->set_precision(PRECISION(kFP16));
  if (flag_bias) {
    param.bias = new Tensor;
    param.bias->Resize({weight_dim[0]});
    param.bias->set_precision(PRECISION(kFP16));
  }
  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.groups = group;
  const float six = 6.f;
  if (flag_act > 0) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type = (paddle::lite_api::ActivationType)
        flag_act;  // 1-relu, 2-relu6, 4-leakyrelu
    if (flag_act == 1) {
      param.fuse_relu = true;
    } else if (flag_act == 2) {
      act_param.Relu_clipped_coef = six;
    } else if (flag_act == 4) {
      act_param.Leaky_relu_alpha = leakey_relu_scale;
    }
    param.activation_param = act_param;
  }

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFP16));

  Tensor filter_fp32;
  filter_fp32.Resize(weight_dim);
  filter_fp32.set_precision(PRECISION(kFloat));
  paddle::lite::fill_tensor_rand(filter_fp32, -1.f, 1.f);
  auto a_ptr = filter_fp32.data<float>();
  auto b_ptr = param.filter->mutable_data<float16_t>();
  for (int i = 0; i < filter_fp32.numel(); i++) {
    b_ptr[i] = static_cast<float16_t>(a_ptr[i]);
  }
  if (flag_bias) {
    Tensor bias_fp32;
    bias_fp32.Resize({weight_dim[0]});
    bias_fp32.set_precision(PRECISION(kFloat));
    paddle::lite::fill_tensor_rand(bias_fp32, -1.f, 1.f);
    a_ptr = bias_fp32.data<float>();
    b_ptr = param.bias->mutable_data<float16_t>();
    for (int i = 0; i < bias_fp32.numel(); i++) {
      b_ptr[i] = static_cast<float16_t>(a_ptr[i]);
    }
  }
  auto wptr = param.filter->data<float16_t>();
  auto bias_ptr = flag_bias ? param.bias->data<float16_t>() : nullptr;

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::ConvCompute<PRECISION(kFP16),
                                              PRECISION(kFP16)>
          conv;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      for (auto& dim_in : input_dims) {
        param.x->Resize(dim_in);
        DDim out_tmp_dims = compute_out_dim(dim_in, param);
        if (out_tmp_dims[2] < 1 || out_tmp_dims[3] < 1) {
          continue;
        }
        param.output->Resize(out_tmp_dims);
        break;
      }
      conv.SetParam(param);
      conv.SetContext(std::move(ctx1));
      /// prepare for run
      conv.PrepareForRun();

      for (auto& dim_in : input_dims) {
        CHECK_EQ(weight_dim[1] * group, dim_in[1])
            << "input channel must equal to weights channel";
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param.x->Resize(dim_in);
        param.output->Resize(dim_out);

        Tensor x_fp32;
        x_fp32.Resize(dim_in);
        x_fp32.set_precision(PRECISION(kFloat));
        paddle::lite::fill_tensor_rand(x_fp32, -1.f, 1.f);
        // paddle::lite::fill_tensor_const(*param.x, 1.f);
        a_ptr = x_fp32.data<float>();
        b_ptr = param.x->mutable_data<float16_t>();
        for (int i = 0; i < x_fp32.numel(); i++) {
          b_ptr[i] = static_cast<float16_t>(a_ptr[i]);
        }
        auto din = param.x->data<float16_t>();

        Tensor tout_basic;
        Tensor tout_basic_fp32;
        if (FLAGS_check_result) {
          tout_basic_fp32.set_precision(PRECISION(kFloat));
          tout_basic.set_precision(PRECISION(kFP16));
          tout_basic_fp32.Resize(dim_out);
          tout_basic.Resize(dim_out);
          auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
          auto dout_basic = tout_basic.mutable_data<float16_t>();

          fill_tensor_const(tout_basic_fp32, 0.f);
          for (int i = 0; i < tout_basic_fp32.numel(); i++) {
            dout_basic[i] = static_cast<float16_t>(dout_basic_fp32[i]);
          }

          conv_basic<float16_t, float16_t>(din,
                                           dout_basic,
                                           dim_in[0],
                                           dim_out[1],
                                           dim_out[2],
                                           dim_out[3],
                                           dim_in[1],
                                           dim_in[2],
                                           dim_in[3],
                                           wptr,
                                           bias_ptr,
                                           group,
                                           weight_dim[3],
                                           weight_dim[2],
                                           strides[1],
                                           strides[0],
                                           dilas[1],
                                           dilas[0],
                                           pads[2],
                                           pads[0],
                                           flag_bias,
                                           flag_act,
                                           six,
                                           leakey_relu_scale);
        }
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          conv.Launch();
        }
        /// compute
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          conv.Launch();
          t0.Stop();
        }

        double gops = 2.0 * dim_out.production() * dim_in[1] * weight_dim[2] *
                      weight_dim[3] / param.groups;
        VLOG(4) << "conv fp32: input shape: " << dim_in << ", output shape"
                << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
                << ", min time: " << t0.LapTimes().Min()
                << ", total GOPS: " << 1e-9 * gops
                << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          auto basic_ptr = tout_basic.data<float16_t>();
          auto saber_ptr = param.output->data<float16_t>();
          paddle::lite::data_diff_kernel(
              basic_ptr, saber_ptr, tout_basic.numel(), max_ratio, max_diff);
          // tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
          VLOG(4) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
          if (std::abs(max_ratio) > 1e-3f) {
            if (max_diff > 5e-4f) {
              int64_t size = tout_basic.numel();
              int64_t width = tout_basic.dims()[tout_basic.dims().size() - 1];
              LOG(WARNING) << "basic result";
              // print_tensor(tout_basic);
              paddle::lite::print_tensor_host_impl(basic_ptr, size, width);
              LOG(WARNING) << "lite result";
              // print_tensor(*param.output);
              paddle::lite::print_tensor_host_impl(saber_ptr, size, width);
              Tensor tdiff;
              tdiff.Resize(tout_basic.dims());
              tdiff.set_precision(PRECISION(kFP16));
              auto ptr = tdiff.mutable_data<float16_t>();
              for (int i = 0; i < size; i++) {
                ptr[i] = saber_ptr[i] - basic_ptr[i];
              }
              auto c_ptr = tdiff.data<float16_t>();
              LOG(WARNING) << "diff result";
              paddle::lite::print_tensor_host_impl(c_ptr, size, width);
              // tensor_diff(tout_basic, *param.output, tdiff);
              // print_tensor(tdiff);
              LOG(FATAL) << "test fp16 conv: input: " << dim_in
                         << ", output: " << dim_out
                         << ", weight dim: " << weight_dim
                         << ", pad: " << pads[0] << ", " << pads[1] << ", "
                         << pads[2] << ", " << pads[3]
                         << ", stride: " << strides[0] << ", " << strides[1]
                         << ", dila_: " << dilas[0] << ", " << dilas[1]
                         << ", group: " << group
                         << ", bias: " << (flag_bias ? "true" : "false")
                         << ", act: " << flag_act << ", threads: " << th
                         << ", power_mode: " << cls << " failed!!\n";
            }
          }
        }
        VLOG(4) << "test fp16 conv: input: " << dim_in
                << ", output: " << dim_out << ", weight dim: " << weight_dim
                << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
                << ", " << pads[3] << ", stride: " << strides[0] << ", "
                << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
                << ", group: " << group
                << ", bias: " << (flag_bias ? "true" : "false")
                << ", act: " << flag_act << ", threads: " << th
                << ", power_mode: " << cls << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.filter;
  delete param.output;
  delete param.bias;
}
#else
void test_conv_fp16(const std::vector<DDim>& input_dims,
                    const DDim& weight_dim,
                    int group,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    int flag_act,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode,
                    const float leakey_relu_scale) {}
#endif  // LITE_WITH_ARM

#if 1  /// conv1x1s1
TEST(TestConv1x1s1, test_conv1x1s1) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 11, 32}) {
      for (auto& cout : {1, 5, 16, 37}) {
        for (auto& g : {1, 2}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_act : {0, 1, 2, 4}) {
              std::vector<DDim> dims;
              if (cin % g != 0 || cout % g != 0) {
                continue;
              }
              DDim weights_dim({cout, cin / g, 1, 1});
              for (auto& batch : {1, 2}) {
                for (auto& h : {1, 7, 19, 28, 32, 56, 1}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                }
              }
              const float leakey_relu_scale = 8.88;
              test_conv_fp16(dims,
                             weights_dim,
                             g,
                             {1, 1},
                             {0, 0, 0, 0},
                             {1, 1},
                             flag_bias,
                             flag_act,
                             {1, 2, 4},
                             {FLAGS_power_mode},
                             leakey_relu_scale);
            }
          }
        }
      }
    }
  }
}
#endif  /// conv1x1s1

#if 0   /// random param conv
TEST(TestConvRand, test_conv_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {1, 5, 16}) {
        for (auto& g : {1, 2}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad_left : {0, 2}) {
                  for (auto& pad_right : {0, 2}) {
                    for (auto& pad_top : {0, 2}) {
                      for (auto& pad_bottom : {0, 2}) {
                        for (auto& dila : {1, 2}) {
                          for (auto& flag_bias : {false, true}) {
                            for (auto& flag_act : {0, 1, 2, 4}) {
                              if (cin % g != 0 || cout % g != 0) {
                                continue;
                              }
                              std::vector<DDim> dims;
                              DDim weights_dim({cout, cin / g, kh, kw});
                              for (auto& batch : {1, 2}) {
                                for (auto& h : {1, 3, 19, 32}) {
                                  dims.push_back(DDim({batch, cin, h, h}));
                                }
                              }
                              // skip 3x3 depthwise conv
                              if (g == cin && cin == cout && kw == 3 &&
                                  kh == 3) {
                                break;
                              }
                              // skip 3x3s1 direct conv
                              if (g == 1 && (cin != 1 || cout != 1) &&
                                  kw == 3 && kh == 3 && stride == 1) {
                                break;
                              }
                              const float leakey_relu_scale = 8.88;
                              test_conv_fp16(
                                  dims,
                                  weights_dim,
                                  g,
                                  {stride, stride},
                                  {pad_top, pad_bottom, pad_left, pad_right},
                                  {dila, dila},
                                  flag_bias,
                                  flag_act,
                                  {4},
                                  {FLAGS_power_mode},
                                  leakey_relu_scale);
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
}
#endif  /// random param conv

#if 1  /// custom
TEST(TestConvCustom, test_conv_fp16_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  test_conv_fp16(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      DDim({FLAGS_out_channel,
            FLAGS_in_channel / FLAGS_group,
            FLAGS_kernel_h,
            FLAGS_kernel_w}),
      FLAGS_group,
      {FLAGS_stride_h, FLAGS_stride_w},
      {FLAGS_pad_h0, FLAGS_pad_h1, FLAGS_pad_w0, FLAGS_pad_w1},
      {FLAGS_dila_h, FLAGS_dila_w},
      FLAGS_flag_bias,
      FLAGS_flag_act,
      {FLAGS_threads},
      {FLAGS_power_mode},
      FLAGS_leakey_relu_alpha);
}
#endif  // custom
