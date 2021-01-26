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
#include "lite/tests/math/conv_ut.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/tensor_utils.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/conv_compute.h"
#endif  // LITE_WITH_ARM

#define CONV_PARAM_INIT                                                 \
  param.strides = strides;                                              \
  param.paddings = std::make_shared<std::vector<int>>(pads);            \
  param.dilations = std::make_shared<std::vector<int>>(dilas);          \
  param.groups = group;                                                 \
  const float six = 6.f;                                                \
  if (flag_act > 0) {                                                   \
    ActivationParam act_param;                                          \
    act_param.has_active = true;                                        \
    /* 1-relu, 2-relu6, 4-leakyrelu */                                  \
    act_param.active_type = (paddle::lite_api::ActivationType)flag_act; \
    if (flag_act == 1) {                                                \
      param.fuse_relu = true;                                           \
    } else if (flag_act == 2) {                                         \
      act_param.Relu_clipped_coef = six;                                \
    } else if (flag_act == 4) {                                         \
      act_param.Leaky_relu_alpha = leakey_relu_scale;                   \
    }                                                                   \
    param.activation_param = act_param;                                 \
  }

#ifdef LITE_WITH_ARM
void test_conv_fp16(const DDim& input_dim,
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
  /*const float six = 6.f;
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
  */
  CONV_PARAM_INIT

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFP16));

  Tensor filter_fp32;
  filter_fp32.Resize(weight_dim);
  filter_fp32.set_precision(PRECISION(kFloat));
  auto a_ptr = filter_fp32.mutable_data<float>();
  auto b_ptr = param.filter->mutable_data<float16_t>();
  fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.filter->numel());
  // fill_data_const<float16_t>(b_ptr, -1.f, param.filter->numel());
  fp16_to_float(param.filter->data<float16_t>(), a_ptr, param.filter->numel());

  Tensor bias_fp32;
  if (flag_bias) {
    bias_fp32.Resize({weight_dim[0]});
    bias_fp32.set_precision(PRECISION(kFloat));
    a_ptr = bias_fp32.mutable_data<float>();
    b_ptr = param.bias->mutable_data<float16_t>();
    fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.bias->numel());
    // fill_data_const<float16_t>(b_ptr, -1.f, param.bias->numel());
    fp16_to_float(param.bias->data<float16_t>(), a_ptr, param.bias->numel());
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
      CHECK_EQ(weight_dim[1] * group, input_dim[1])
          << "input channel must equal to weights channel";
      DDim dim_out = compute_out_dim(input_dim, param);
      if (dim_out[2] < 1 || dim_out[3] < 1) {
        return;
      }
      param.output->Resize(dim_out);
      conv.SetParam(param);
      conv.SetContext(std::move(ctx1));
      /// prepare for run
      conv.PrepareForRun();
      // param init
      param.x->Resize(input_dim);
      param.output->Resize(dim_out);

      // set input data
      Tensor x_fp32;
      x_fp32.Resize(input_dim);
      x_fp32.set_precision(PRECISION(kFloat));
      a_ptr = x_fp32.mutable_data<float>();
      b_ptr = param.x->mutable_data<float16_t>();
      fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.x->numel());
      // fill_data_const<float16_t>(b_ptr, -1.f, param.x->numel());
      fp16_to_float(param.x->data<float16_t>(), a_ptr, param.x->numel());
      auto din = param.x->data<float16_t>();
      auto din_fp32 = x_fp32.data<float>();

      Tensor tout_basic;
      Tensor tout_basic_fp32;
      Tensor tout_basic_fp16;
      Tensor tout_basic_diff;
      double basic_max_ratio = 0;
      double basic_max_diff = 0;
      if (FLAGS_check_result) {
        tout_basic_fp32.set_precision(PRECISION(kFloat));
        tout_basic.set_precision(PRECISION(kFP16));
        tout_basic_fp16.set_precision(PRECISION(kFP16));
        tout_basic_diff.set_precision(PRECISION(kFP16));
        tout_basic_fp32.Resize(dim_out);
        tout_basic.Resize(dim_out);
        tout_basic_fp16.Resize(dim_out);
        tout_basic_diff.Resize(dim_out);
        auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
        auto dout_basic = tout_basic.mutable_data<float16_t>();
        auto bias_fp32_ptr = flag_bias ? bias_fp32.data<float>() : nullptr;
        auto filter_fp32_ptr = filter_fp32.data<float>();

        fill_data_const<float>(dout_basic_fp32, 0.f, tout_basic_fp32.numel());
        fill_data_const<float16_t>(dout_basic, 0.f, tout_basic.numel());

        conv_basic<float, float>(din_fp32,
                                 dout_basic_fp32,
                                 input_dim[0],
                                 dim_out[1],
                                 dim_out[2],
                                 dim_out[3],
                                 input_dim[1],
                                 input_dim[2],
                                 input_dim[3],
                                 filter_fp32_ptr,
                                 bias_fp32_ptr,
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
        conv_basic<float16_t, float16_t>(din,
                                         dout_basic,
                                         input_dim[0],
                                         dim_out[1],
                                         dim_out[2],
                                         dim_out[3],
                                         input_dim[1],
                                         input_dim[2],
                                         input_dim[3],
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
        // fp32 -> fp16
        auto dout_basic_fp16_ptr = tout_basic_fp16.mutable_data<float16_t>();
        auto diff_ptr = tout_basic_diff.mutable_data<float16_t>();
        float_to_fp16(
            dout_basic_fp32, dout_basic_fp16_ptr, tout_basic_fp16.numel());
        // basic_diff: fp16 - (fp32->fp16)
        data_diff(dout_basic,
                  dout_basic_fp16_ptr,
                  diff_ptr,
                  tout_basic.numel(),
                  basic_max_ratio,
                  basic_max_diff);
        // VLOG(4) << "compare result, max diff: " << basic_max_diff
        //         << ", max ratio: " << basic_max_ratio;
        VLOG_PRINT_DIFF(basic_max_diff, basic_max_ratio)
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

      double gops = 2.0 * dim_out.production() * input_dim[1] * weight_dim[2] *
                    weight_dim[3] / param.groups;
      // VLOG(4) << "conv fp32: input shape: " << input_dim << ", output shape"
      //         << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
      //         << ", min time: " << t0.LapTimes().Min()
      //         << ", total GOPS: " << 1e-9 * gops
      //         << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
      //         << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();
      VLOG_PRINT_GOPS(input_dim, dim_out, t0, gops)

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        auto basic_ptr = tout_basic_fp16.data<float16_t>();
        auto saber_ptr = param.output->data<float16_t>();
        Tensor tdiff;
        tdiff.Resize(tout_basic.dims());
        tdiff.set_precision(PRECISION(kFP16));
        auto ptr = tdiff.mutable_data<float16_t>();
        // paddle::lite::data_diff_kernel(
        data_diff(
            basic_ptr, saber_ptr, ptr, tout_basic.numel(), max_ratio, max_diff);
        // tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
        // VLOG(4) << "compare result, max diff: " << max_diff
        //         << ", max ratio: " << max_ratio;
        VLOG_PRINT_DIFF(max_diff, max_ratio)
        if (max_diff > basic_max_diff) {
          int64_t size = tout_basic.numel();
          int count = 0;
          bool check = true;
          for (int i = 0; i < size; i++) {
            if (abs(ptr[i]) > 1) {
              check = false;
              break;
            }
            if (abs(ptr[i]) > 0.01) {
              count += 1;
            }
          }
          VLOG(4) << "check: " << check << ", count: " << count;
          check = check && count < std::max(10, static_cast<int>(0.01 * size));
          if (!check) {
            int64_t width = tout_basic.dims()[tout_basic.dims().size() - 1];
            /*LOG(WARNING) << "basic result";
            print_tensor(basic_ptr, size, width);
            LOG(WARNING) << "lite result";
            // print_tensor(*param.output);
            print_tensor(saber_ptr, size, width);
            LOG(WARNING) << "diff result";
            print_tensor(ptr, size, width);
            LOG(FATAL) << "test fp16 conv: input: " << input_dim
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
            */
            VLOG_FAIL_INFO(basic_ptr, saber_ptr, ptr, size, width)
            VLOG_FAILED_INFO(input_dim, dim_out)
          }
        }
      }
      /*VLOG(4) << "test fp16 conv: input: " << input_dim
              << ", output: " << dim_out << ", weight dim: " << weight_dim
              << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
              << ", " << pads[3] << ", stride: " << strides[0] << ", "
              << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
              << ", group: " << group
              << ", bias: " << (flag_bias ? "true" : "false")
              << ", act: " << flag_act << ", threads: " << th
              << ", power_mode: " << cls << " successed!!\n";
       */
      VLOG_SUCCESSED_INFO(input_dim, dim_out)
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
              if (cin % g != 0 || cout % g != 0) {
                continue;
              }
              DDim weights_dim({cout, cin / g, 1, 1});
              for (auto& batch : {1, 2}) {
                for (auto& h : {1, 7, 19, 28, 32, 56, 1}) {
                  DDim in_dim({batch, cin, h, h});
                  const float leakey_relu_scale = 1.0f;
                  test_conv_fp16(in_dim,
                                 weights_dim,
                                 g,
                                 {1, 1},
                                 {0, 0, 0, 0},
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
#endif  /// conv1x1s1

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
                    if (cin == 1 && cout == 1) {
                      continue;
                    }
                    DDim weights_dim({cout, cin, 3, 3});
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {3, 7, 15, 56, 32}) {
                        DDim in_dim({batch, cin, h, h});
                        const float leakey_relu_scale = 1.0f;
                        test_conv_fp16(in_dim,
                                       weights_dim,
                                       g,
                                       {1, 1},
                                       {0, 0, 0, 0},
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
                                  dims.clear();
                                  dims.push_back(DDim({batch, cin, h, h}));
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
                                  const float leakey_relu_scale = 1.0f;
                                  test_conv_fp16(dims,
                                                 weights_dim,
                                                 g,
                                                 {stride, stride},
                                                 {pad_top,
                                                  pad_bottom,
                                                  pad_left,
                                                  pad_right},
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
