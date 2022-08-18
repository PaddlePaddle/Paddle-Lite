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
DEFINE_bool(basic_test, false, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 112, "input height");
DEFINE_int32(in_width, 112, "input width");

DEFINE_int32(out_channel, 32, "output channel");
DEFINE_int32(group, 32, "group");
DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h, 1, "pad height");
DEFINE_int32(pad_w, 1, "pad width");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_int32(flag_act, 1, "do act");
DEFINE_bool(flag_bias, true, "with bias");
DEFINE_double(clipped_coef, 1.0, "clipped relu coef");
DEFINE_double(leakey_relu_alpha, 2.22, "leakey relu alpha");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ConvParam ConvParam;
typedef paddle::lite::operators::ActivationParam ActivationParam;
using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::ConvParam& param) {
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  DDim dim_out = dim_in;
  dim_out[1] = param.filter->dims()[0];
  auto kernel_h = param.filter->dims()[2];
  auto kernel_w = param.filter->dims()[3];
  auto h = dim_in[2];
  auto w = dim_in[3];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  auto hout = (h + paddings[0] + paddings[1] - kernel_exten) / stride_h + 1;
  kernel_exten = dila_w * (kernel_w - 1) + 1;
  auto wout = (w + paddings[2] + paddings[3] - kernel_exten) / stride_w + 1;
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

template <paddle::lite::PrecisionType ptype>
void get_conv_param(const DDim& dim_w,
                    int g,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dila,
                    bool flag_bias,
                    bool flag_relu,
                    ConvParam* param) {
  param->x = new Tensor;
  param->x->set_precision(PRECISION(kInt8));
  param->filter = new Tensor;
  param->filter->Resize(dim_w);
  param->filter->set_precision(PRECISION(kInt8));
  if (flag_bias) {
    param->bias = new Tensor;
    param->bias->Resize({dim_w[0]});
    param->bias->set_precision(PRECISION(kFloat));
  }
  param->strides = strides;
  param->paddings = std::make_shared<std::vector<int>>(pads);
  param->dilations = std::make_shared<std::vector<int>>(dila);
  param->fuse_relu = flag_relu;
  param->groups = g;

  param->output = new Tensor;
  param->output->set_precision(ptype);
}

void release_param(ConvParam* param) {
  delete param->x;
  delete param->filter;
  delete param->output;
  delete param->bias;
}

#ifdef LITE_WITH_ARM
#include "lite/backends/arm/math/funcs.h"
void test_conv_int8(const std::vector<DDim>& input_dims,
                    const DDim& weight_dim,
                    int group,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    int flag_act,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode,
                    const float six = 6.f,
                    const float alpha = 1.f) {
  paddle::lite::DeviceInfo::Init();
  ConvParam param_int8_out;
  ConvParam param_fp32_out;

  get_conv_param<PRECISION(kInt8)>(weight_dim,
                                   group,
                                   strides,
                                   pads,
                                   dilas,
                                   flag_bias,
                                   flag_act > 0,
                                   &param_int8_out);

  get_conv_param<PRECISION(kFloat)>(weight_dim,
                                    group,
                                    strides,
                                    pads,
                                    dilas,
                                    flag_bias,
                                    flag_act > 0,
                                    &param_fp32_out);
  Tensor weight_fp32;
  Tensor bias_fp32;
  weight_fp32.Resize(weight_dim);
  paddle::lite::fill_tensor_rand(*param_int8_out.filter, -127, 127);
  param_fp32_out.filter->CopyDataFrom(*param_int8_out.filter);
  if (flag_bias) {
    auto dim_b = param_int8_out.bias->dims();
    bias_fp32.Resize(dim_b);
    paddle::lite::fill_tensor_rand(*param_int8_out.bias, -1.f, 1.f);
    param_fp32_out.bias->CopyDataFrom(*param_int8_out.bias);
    bias_fp32.CopyDataFrom(*param_int8_out.bias);
  }
  if (flag_act > 0) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type = (paddle::lite_api::ActivationType)
        flag_act;  // 1-relu, 2-relu6, 4-leakyrelu
    if (flag_act == 1) {
      param_fp32_out.fuse_relu = true;
      param_int8_out.fuse_relu = true;
    } else if (flag_act == 2) {
      act_param.Relu_clipped_coef = six;
    } else if (flag_act == 4) {
      act_param.Leaky_relu_alpha = alpha;
    }
    param_fp32_out.activation_param = act_param;
    param_int8_out.activation_param = act_param;
  }

  std::vector<float> scale_in{1.f / 127};
  std::vector<float> scale_out(1, weight_dim.count(1, 4) / 127.f);
  if (flag_act == 2) {
    scale_out[0] = six / 127.f;
  } else if (flag_act == 4) {
    if (std::abs(alpha) > 1) {
      scale_out[0] *= std::abs(alpha);
    }
  }
  std::vector<float> scale_w(weight_dim[0], 1.f / 127);

  param_int8_out.input_scale = scale_in[0];
  param_int8_out.output_scale = scale_out[0];
  param_int8_out.weight_scale = scale_w;

  param_fp32_out.input_scale = scale_in[0];
  param_fp32_out.output_scale = scale_out[0];
  param_fp32_out.weight_scale = scale_w;

  auto wptr_fp32 = weight_fp32.mutable_data<float>();
  auto bptr_fp32 = flag_bias ? bias_fp32.data<float>() : nullptr;

  paddle::lite::arm::math::int8_to_fp32(param_int8_out.filter->data<int8_t>(),
                                        wptr_fp32,
                                        scale_w.data(),
                                        weight_dim[0],
                                        1,
                                        weight_dim.count(1, 4));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      std::unique_ptr<paddle::lite::KernelContext> ctx2(
          new paddle::lite::KernelContext);
      auto& ctx_tmp1 = ctx1->As<paddle::lite::ARMContext>();
      ctx_tmp1.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      auto& ctx_tmp2 = ctx2->As<paddle::lite::ARMContext>();
      ctx_tmp2.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);

      paddle::lite::kernels::arm::ConvCompute<PRECISION(kInt8),
                                              PRECISION(kInt8)>
          conv_int8_int8;
      paddle::lite::kernels::arm::ConvCompute<PRECISION(kInt8),
                                              PRECISION(kFloat)>
          conv_int8_fp32;
      conv_int8_int8.SetContext(std::move(ctx1));
      conv_int8_fp32.SetContext(std::move(ctx2));

      /// set param and context

      for (auto& dim_in : input_dims) {
        param_int8_out.x->Resize(dim_in);
        DDim out_tmp_dims = compute_out_dim(dim_in, param_int8_out);
        if (out_tmp_dims[2] < 1 || out_tmp_dims[3] < 1) {
          return;
        }
        param_fp32_out.x->Resize(dim_in);
        param_int8_out.output->Resize(out_tmp_dims);
        param_fp32_out.output->Resize(out_tmp_dims);
        break;
      }

      conv_int8_int8.SetParam(param_int8_out);
      conv_int8_fp32.SetParam(param_fp32_out);
      /// prepare for run
      conv_int8_int8.PrepareForRun();
      conv_int8_fp32.PrepareForRun();

      for (auto& dim_in : input_dims) {
        CHECK_EQ(weight_dim[1] * group, dim_in[1])
            << "input channel must equal to weights channel";
        DDim dim_out = compute_out_dim(dim_in, param_int8_out);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param_fp32_out.output->set_precision(PRECISION(kFloat));
        param_int8_out.output->set_precision(PRECISION(kInt8));

        param_int8_out.x->Resize(dim_in);
        param_int8_out.output->Resize(dim_out);
        param_fp32_out.x->Resize(dim_in);
        param_fp32_out.output->Resize(dim_out);

        Tensor tin_fp32;
        tin_fp32.Resize(dim_in);
        tin_fp32.set_precision(PRECISION(kFloat));
        Tensor tout_basic_fp32;
        Tensor tout_basic_int8;

        paddle::lite::fill_tensor_rand(*param_int8_out.x, -127, 127);
        param_fp32_out.x->CopyDataFrom(*param_int8_out.x);

        auto din_fp32 = tin_fp32.mutable_data<float>();
        paddle::lite::arm::math::int8_to_fp32(param_int8_out.x->data<int8_t>(),
                                              din_fp32,
                                              scale_in.data(),
                                              1,
                                              1,
                                              dim_in.production());

        if (FLAGS_check_result) {
          tout_basic_fp32.set_precision(PRECISION(kFloat));
          tout_basic_fp32.Resize(dim_out);
          tout_basic_int8.set_precision(PRECISION(kInt8));
          tout_basic_int8.Resize(dim_out);
          fill_tensor_const(tout_basic_fp32, 0.f);
          auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
          auto dout_basic_int8 = tout_basic_int8.mutable_data<int8_t>();
          conv_basic<float, float>(din_fp32,
                                   dout_basic_fp32,
                                   dim_in[0],
                                   dim_out[1],
                                   dim_out[2],
                                   dim_out[3],
                                   dim_in[1],
                                   dim_in[2],
                                   dim_in[3],
                                   wptr_fp32,
                                   bptr_fp32,
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
                                   alpha);
          paddle::lite::arm::math::fp32_to_int8(dout_basic_fp32,
                                                dout_basic_int8,
                                                scale_out.data(),
                                                1,
                                                1,
                                                dim_out.production());
        }
        double gops = 2.0 * dim_out.production() * dim_in[1] * weight_dim[2] *
                      weight_dim[3] / group;
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          conv_int8_fp32.Launch();
        }
        /// compute fp32 output
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          conv_int8_fp32.Launch();
          t0.Stop();
        }

        LOG(INFO) << "int8 conv, fp32 output: output shape" << dim_out
                  << ",running time, avg: " << t0.LapTimes().Avg() << " ms"
                  << ", min time: " << t0.LapTimes().Min() << " ms"
                  << ", total GOPS: " << 1e-9 * gops
                  << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                  << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

        // compute int8 output
        t0.Reset();
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          conv_int8_int8.Launch();
          t0.Stop();
        }

        LOG(INFO) << "int8 conv, int8 output: output shape" << dim_out
                  << ",running time, avg: " << t0.LapTimes().Avg()
                  << ", min time: " << t0.LapTimes().Min()
                  << ", total GOPS: " << 1e-9 * gops
                  << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                  << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();
        /// compare result fp32 output
        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          tensor_cmp_host(
              tout_basic_fp32, *param_fp32_out.output, max_ratio, max_diff);
          LOG(INFO) << "FP32 compare result, max diff: " << max_diff
                    << ", max ratio: " << max_ratio;
          if (std::abs(max_ratio) > 1e-5f) {
            if (max_diff > 5e-5f) {
              LOG(WARNING) << "basic result";
              print_tensor(tout_basic_fp32);
              LOG(WARNING) << "lite result";
              print_tensor(*param_fp32_out.output);
              Tensor tdiff;
              tdiff.Resize(tout_basic_fp32.dims());
              tdiff.set_precision(PRECISION(kFloat));
              tensor_diff(tout_basic_fp32, *param_fp32_out.output, tdiff);
              print_tensor(tdiff);
              release_param(&param_int8_out);
              release_param(&param_fp32_out);
              LOG(FATAL) << "test int8 conv, fp32 out: input: " << dim_in
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
        // compare result int8 output
        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          // ! int8
          tensor_cmp_host(
              tout_basic_int8, *param_int8_out.output, max_ratio, max_diff);
          LOG(INFO) << "int8 compare result, max diff: " << max_diff
                    << ", max ratio: " << max_ratio;
          if (fabs(max_diff) > 0) {
            Tensor tdiff;
            tdiff.Resize(tout_basic_int8.dims());
            tdiff.set_precision(PRECISION(kInt8));
            tensor_diff(tout_basic_int8, *param_int8_out.output, tdiff);
            auto ptr = tdiff.data<int8_t>();
            auto ptr_basic_fp32 = tout_basic_fp32.data<float>();
            float count = 0;
            bool check = true;
            for (int i = 0; i < tdiff.numel(); ++i) {
              if (abs(ptr[i]) > 1) {
                check = false;
                LOG(ERROR) << "basic float data: " << ptr_basic_fp32[i]
                           << ", after scale: "
                           << ptr_basic_fp32[i] / scale_out[0];
                break;
              }
              if (ptr[i] != 0) {
                LOG(ERROR) << "basic float data: " << ptr_basic_fp32[i]
                           << ", after scale: "
                           << ptr_basic_fp32[i] / scale_out[0];
                count += 1;
              }
            }
            check =
                check &&
                count < std::max(10, static_cast<int>(0.01 * tdiff.numel()));
            if (!check) {
              LOG(WARNING) << "int8 basic result";
              print_tensor(tout_basic_int8);
              LOG(WARNING) << "int8 lite result";
              print_tensor(*param_int8_out.output);
              LOG(WARNING) << "int8 diff tensor";
              print_tensor(tdiff);
              release_param(&param_int8_out);
              release_param(&param_fp32_out);
              LOG(FATAL) << "test int8 conv, int8 out: input: " << dim_in
                         << ", output: " << dim_out
                         << ", weight dim: " << weight_dim
                         << ", pad: " << pads[0] << ", " << pads[1] << ", "
                         << pads[2] << ", " << pads[3]
                         << ", stride: " << strides[0] << ", " << strides[1]
                         << ", dila_: " << dilas[0] << ", " << dilas[1]
                         << ", bias: " << (flag_bias ? "true" : "false")
                         << ", act: " << flag_act << ", threads: " << th
                         << ", power_mode: " << cls << " failed!!\n";
            }
          }
        }
        LOG(INFO) << "test int8 conv: input: " << dim_in
                  << ", output: " << dim_out << ", weight dim: " << weight_dim
                  << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
                  << ", " << pads[3] << ", stride: " << strides[0] << ", "
                  << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
                  << ", bias: " << (flag_bias ? "true" : "false")
                  << ", act: " << flag_act << ", threads: " << th
                  << ", power_mode: " << cls << " successed!!\n";
      }
    }
  }
  release_param(&param_int8_out);
  release_param(&param_fp32_out);
}
#else
void test_conv_int8(const std::vector<DDim>& input_dims,
                    const DDim& weight_dim,
                    int group,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    int flag_act,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode,
                    float six = 6.f,
                    float alpha = 1.f) {}
#endif  // LITE_WITH_ARM

#if 1  /// 3x3dw
TEST(TestConv3x3DWInt8, test_conv3x3_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride : {1, 2}) {
      for (auto& pad : {0, 1}) {
        for (auto& flag_bias : {false, true}) {
          for (auto& flag_act : {0, 1, 2, 4}) {
            for (auto& c : {1, 3, 5, 8, 16, 32}) {
              std::vector<DDim> dims;
              DDim weights_dim({c, 1, 3, 3});
              for (auto& batch : {1, 2}) {
                for (auto& h : {33, 1, 15, 3}) {
                  dims.push_back(DDim({batch, c, h, h}));
                }
              }
              test_conv_int8(dims,
                             weights_dim,
                             c,
                             {stride, stride},
                             {pad, pad, pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_act,
                             {4},
                             {FLAGS_power_mode},
                             FLAGS_clipped_coef,
                             FLAGS_leakey_relu_alpha);
            }
          }
        }
      }
    }
  }
}
#endif  /// 3x3dw

#if 1  /// 5x5dw
TEST(TestConv5x5DWInt8, test_conv5x5_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride : {1, 2}) {
      for (auto& pad : {0, 1, 2, 3, 4}) {
        for (auto& flag_bias : {false, true}) {
          for (auto& flag_act : {0, 1, 2, 4}) {
            for (auto& c : {1, 5, 15, 33}) {
              DDim weights_dim({c, 1, 5, 5});
              std::vector<DDim> dims;
              for (auto& batch : {1, 2}) {
                for (auto& h : {224, 1, 112, 3, 33, 15}) {
                  dims.push_back(DDim({batch, c, h, h}));
                }
              }
              test_conv_int8(dims,
                             weights_dim,
                             c,
                             {stride, stride},
                             {pad, pad, pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_act,
                             {4},
                             {FLAGS_power_mode},
                             FLAGS_clipped_coef,
                             FLAGS_leakey_relu_alpha);
            }
          }
        }
      }
    }
  }
}
#endif  /// 5x5dw

#if 1  /// conv1x1s1
TEST(TestConv1x1s1Int8, test_conv1x1s1) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 33}) {
      for (auto& cout : {1, 5, 17}) {
        for (auto& g : {1, 2}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_act : {0, 1, 2, 4}) {
              if (cin % g != 0 || cout % g != 0) {
                continue;
              }

              DDim weights_dim({cout, cin / g, 1, 1});
              std::vector<DDim> dims;
              for (auto& batch : {1, 2}) {
                for (auto& h : {33, 1, 16, 9}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                }
              }
              test_conv_int8(dims,
                             weights_dim,
                             g,
                             {1, 1},
                             {0, 0, 0, 0},
                             {1, 1},
                             flag_bias,
                             flag_act,
                             {4},
                             {FLAGS_power_mode},
                             FLAGS_clipped_coef,
                             FLAGS_leakey_relu_alpha);
            }
          }
        }
      }
    }
  }
}
#endif  /// conv1x1s1

#if 1  /// conv3x3s1
TEST(TestConv3x3s1Int8, test_conv_3x3s1) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 33}) {
      for (auto& cout : {1, 5, 33}) {
        for (auto& pad_top : {1, 2}) {
          for (auto& pad_bottom : {1, 2}) {
            for (auto& pad_left : {1, 2}) {
              for (auto& pad_right : {1, 2}) {
                for (auto& flag_bias : {false, true}) {
                  for (auto& flag_act : {0, 1}) {
                    DDim weights_dim({cout, cin, 3, 3});
                    std::vector<DDim> dims;
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {33, 1, 17, 7}) {
                        if (cin == 1 && cout == 1) {
                          continue;
                        }
                        dims.push_back(DDim({batch, cin, h, h}));
                      }
                    }
                    if (dims.size() != 0) {
                      test_conv_int8(dims,
                                     weights_dim,
                                     1,
                                     {1, 1},
                                     {pad_top, pad_bottom, pad_left, pad_right},
                                     {1, 1},
                                     flag_bias,
                                     flag_act,
                                     {4},
                                     {FLAGS_power_mode},
                                     FLAGS_clipped_coef,
                                     FLAGS_leakey_relu_alpha);
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
#endif  /// conv3x3s1

#if 1  /// conv3x3s2
TEST(TestConv3x3s2Int8, test_conv_3x3s2) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 31}) {
      for (auto& cout : {1, 5, 33}) {
        for (auto& pad_top : {1, 2}) {
          for (auto& pad_bottom : {1, 2}) {
            for (auto& pad_left : {1, 2}) {
              for (auto& pad_right : {1, 2}) {
                for (auto& flag_bias : {false, true}) {
                  for (auto& flag_act : {0, 1, 2, 4}) {
                    DDim weights_dim({cout, cin, 3, 3});
                    std::vector<DDim> dims;
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {33, 1, 19, 7, 33}) {
                        dims.push_back(DDim({batch, cin, h, h}));
                      }
                    }
                    test_conv_int8(dims,
                                   weights_dim,
                                   1,
                                   {2, 2},
                                   {pad_top, pad_bottom, pad_left, pad_right},
                                   {1, 1},
                                   flag_bias,
                                   flag_act,
                                   {4},
                                   {FLAGS_power_mode},
                                   FLAGS_clipped_coef,
                                   FLAGS_leakey_relu_alpha);
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
TEST(TestConvRandInt8, test_conv_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 17}) {
      for (auto& cout : {1, 8, 17}) {
        for (auto& g : {1, 2}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad_top : {0, 1, 2}) {
                  for (auto& pad_bottom : {0, 1, 2}) {
                    for (auto& pad_left : {0, 1, 2}) {
                      for (auto& pad_right : {0, 1, 2}) {
                        for (auto& dila : {1, 2}) {
                          for (auto& flag_bias : {false, true}) {
                            for (auto& flag_act : {0, 1, 2, 4}) {
                              if (cin % g != 0 || cout % g != 0) {
                                break;
                              }
                              DDim weights_dim({cout, cin / g, kh, kw});
                              std::vector<DDim> dims;
                              for (auto& batch : {1, 2}) {
                                for (auto& h : {1, 64, 3, 18, 5, 19}) {
                                  dims.push_back(DDim({batch, cin, h, h}));
                                }
                              }
                              test_conv_int8(
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
                                  FLAGS_clipped_coef,
                                  FLAGS_leakey_relu_alpha);
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
TEST(TestConvCustomInt8, test_conv_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  std::vector<DDim> dims;
  dims.push_back(
      DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width}));
  test_conv_int8(dims,
                 DDim({FLAGS_out_channel,
                       FLAGS_in_channel / FLAGS_group,
                       FLAGS_kernel_h,
                       FLAGS_kernel_w}),
                 FLAGS_group,
                 {FLAGS_stride_h, FLAGS_stride_w},
                 {FLAGS_pad_h, FLAGS_pad_h, FLAGS_pad_w, FLAGS_pad_w},
                 {FLAGS_dila_h, FLAGS_dila_w},
                 FLAGS_flag_bias,
                 FLAGS_flag_act,
                 {FLAGS_threads},
                 {FLAGS_power_mode},
                 FLAGS_clipped_coef,
                 FLAGS_leakey_relu_alpha);
}
#endif  // custom

#ifdef LITE_WITH_ARM8_SVE2  /// conv3x3s2
TEST(TestConv3x3s2Int8SVE2, test_conv_3x3s2_sve2) {
  if (1) {
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {5, 16, 32}) {
        for (auto& pad_top : {1, 0}) {
          for (auto& pad_bottom : {pad_top}) {
            for (auto& pad_left : {1, 0}) {
              for (auto& pad_right : {pad_left}) {
                for (auto& flag_bias : {false, true}) {
                  for (auto& flag_act : {0, 1, 2, 4}) {
                    DDim weights_dim({cout, cin, 3, 3});
                    std::vector<DDim> dims;
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {3, 7, 9, 15, 39, 41}) {
                        dims.push_back(DDim({batch, cin, h, h}));
                      }
                    }
                    test_conv_int8(dims,
                                   weights_dim,
                                   1,
                                   {2, 2},
                                   {pad_top, pad_bottom, pad_left, pad_right},
                                   {1, 1},
                                   flag_bias,
                                   flag_act,
                                   {4},
                                   {FLAGS_power_mode},
                                   FLAGS_clipped_coef,
                                   FLAGS_leakey_relu_alpha);
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
