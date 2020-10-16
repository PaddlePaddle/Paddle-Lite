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
void test_conv_fp32(const std::vector<DDim>& input_dims,
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
  param.x->set_precision(PRECISION(kFloat));
  param.filter = new Tensor;
  param.filter->Resize(weight_dim);
  param.filter->set_precision(PRECISION(kFloat));
  if (flag_bias) {
    param.bias = new Tensor;
    param.bias->Resize({weight_dim[0]});
    param.bias->set_precision(PRECISION(kFloat));
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
  param.output->set_precision(PRECISION(kFloat));

  paddle::lite::fill_tensor_rand(*param.filter, -1.f, 1.f);
  //  paddle::lite::fill_tensor_const(*param.filter, 1.f);
  if (flag_bias) {
    paddle::lite::fill_tensor_rand(*param.bias, -1.f, 1.f);
    //    paddle::lite::fill_tensor_const(*param.bias, 1.f);
  }
  auto wptr = param.filter->data<float>();
  auto bias_ptr = flag_bias ? param.bias->data<float>() : nullptr;

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::ConvCompute<PRECISION(kFloat),
                                              PRECISION(kFloat)>
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

        paddle::lite::fill_tensor_rand(*param.x, -1.f, 1.f);
        // paddle::lite::fill_tensor_const(*param.x, 1.f);
        auto din = param.x->data<float>();

        Tensor tout_basic;
        if (FLAGS_check_result) {
          tout_basic.set_precision(PRECISION(kFloat));
          tout_basic.Resize(dim_out);
          fill_tensor_const(tout_basic, 0.f);
          auto dout_basic = tout_basic.mutable_data<float>();
          conv_basic<float, float>(din,
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
          tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
          VLOG(4) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
          if (std::abs(max_ratio) > 1e-3f) {
            if (max_diff > 5e-4f) {
              LOG(WARNING) << "basic result";
              print_tensor(tout_basic);
              LOG(WARNING) << "lite result";
              print_tensor(*param.output);
              Tensor tdiff;
              tdiff.Resize(tout_basic.dims());
              tdiff.set_precision(PRECISION(kFloat));
              tensor_diff(tout_basic, *param.output, tdiff);
              print_tensor(tdiff);
              LOG(FATAL) << "test fp32 conv: input: " << dim_in
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
        VLOG(4) << "test fp32 conv: input: " << dim_in
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
void test_conv_fp32(const std::vector<DDim>& input_dims,
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

#if 0   // 3x3dw if only run one case. its ok
TEST(TestConv3x3DW, test_conv3x3_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride : {1, 2}) {
      for (auto& pad_left : {0, 1, 2}) {
        for (auto& pad_right : {0, 1, 2}) {
          for (auto& pad_top : {0, 1, 2}) {
            for (auto& pad_bottom : {0, 1, 2}) {
              for (auto& flag_bias : {false, true}) {
                for (auto& flag_act : {0, 1, 2, 4}) {
                  for (auto& c : {1, 3, 5, 8, 16, 32}) {
                    std::vector<DDim> dims;
                    DDim weights_dim({c, 1, 3, 3});
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {1, 3, 15, 19, 28, 32, 75}) {
                        dims.push_back(DDim({batch, c, h, h}));
                      }
                    }
                    const float leakey_relu_scale = 8.88;
                    test_conv_fp32(dims,
                                   weights_dim,
                                   c,
                                   {stride, stride},
                                   {pad_top, pad_bottom, pad_left, pad_right},
                                   {1, 1},
                                   flag_bias,
                                   flag_act,
                                   {1},
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
#endif  /// 3x3dw

#if 1  /// 5x5dw
TEST(TestConv5x5DW, test_conv5x5_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride : {1, 2}) {
      for (auto& pad_left : {0, 1, 2}) {
        for (auto& pad_right : {0, 1, 2}) {
          for (auto& pad_top : {0, 1, 2}) {
            for (auto& pad_bottom : {0, 1, 2}) {
              for (auto& flag_bias : {false, true}) {
                for (auto& flag_act : {0, 1, 2, 4}) {
                  for (auto& c : {1, 15, 32}) {
                    std::vector<DDim> dims;
                    DDim weights_dim({c, 1, 5, 5});
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {1, 3, 15, 56}) {
                        dims.push_back(DDim({batch, c, h, h}));
                      }
                    }
                    const float leakey_relu_scale = 8.88;
                    test_conv_fp32(dims,
                                   weights_dim,
                                   c,
                                   {stride, stride},
                                   {pad_left, pad_right, pad_top, pad_bottom},
                                   {1, 1},
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
#endif  /// 5x5dw

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
              test_conv_fp32(dims,
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

// TODO(MyPandaShaoxiang): fix me, diff: 3x3s1 winograd
#if 0   /// conv3x3s1
TEST(TestConv3x3s1, test_conv_3x3s1) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 8}) {
      for (auto& cout : {1, 5, 32, 48}) {
        for (auto& pad_left : {0, 1, 2}) {
          for (auto& pad_right : {0, 1, 2}) {
            for (auto& pad_top : {0, 1, 2}) {
              for (auto& pad_bottom : {0, 1, 2}) {
                for (auto& flag_bias : {false, true}) {
                  for (auto& flag_act : {0, 1, 2, 4}) {
                    std::vector<DDim> dims;
                    DDim weights_dim({cout, cin, 3, 3});
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {1, 3, 17, 33}) {
                        dims.push_back(DDim({batch, cin, h, h}));
                      }
                    }
                    if (cin == 1 && cout == 1) {
                      continue;
                    }
                    const float leakey_relu_scale = 8.88;
                    test_conv_fp32(dims,
                                   weights_dim,
                                   1,
                                   {1, 1},
                                   {pad_top, pad_bottom, pad_left, pad_right},
                                   {1, 1},
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
#endif  /// conv3x3s1

#if 1  /// conv3x3s2
TEST(TestConv3x3s2, test_conv_3x3s2) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {1, 3, 9, 32}) {
        for (auto& pad_left : {0, 1, 2}) {
          for (auto& pad_right : {0, 1, 2}) {
            for (auto& pad_top : {0, 1, 2}) {
              for (auto& pad_bottom : {0, 1, 2}) {
                for (auto& flag_bias : {false, true}) {
                  for (auto& flag_act : {0, 1, 2, 4}) {
                    std::vector<DDim> dims;
                    DDim weights_dim({cout, cin, 3, 3});
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {3, 7, 15, 56, 32}) {
                        dims.push_back(DDim({batch, cin, h, h}));
                      }
                    }
                    if (cin == 1 && cout == 1) {
                      continue;
                    }
                    const float leakey_relu_scale = 8.88;
                    test_conv_fp32(dims,
                                   weights_dim,
                                   1,
                                   {2, 2},
                                   {pad_top, pad_bottom, pad_left, pad_right},
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
    }
  }
}
#endif  /// conv3x3s2

#if 1  /// random param conv
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
                              test_conv_fp32(
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
TEST(TestConvCustom, test_conv_fp32_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  test_conv_fp32(
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
