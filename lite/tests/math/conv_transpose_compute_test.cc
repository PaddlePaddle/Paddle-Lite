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
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/print_info.h"
#include "lite/tests/utils/tensor_utils.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/conv_transpose_compute.h"
#endif  // LITE_WITH_ARM
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

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

#if defined(LITE_WITH_ARM)
DEFINE_bool(basic_test, true, "do all tests");
#else
DEFINE_bool(basic_test, false, "do all tests");
#endif
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 32, "input height");
DEFINE_int32(in_width, 32, "input width");

DEFINE_int32(out_channel, 64, "output channel");
DEFINE_int32(group, 1, "group");
DEFINE_int32(kernel_h, 2, "kernel height");
DEFINE_int32(kernel_w, 2, "kernel width");
DEFINE_int32(pad_h, 0, "pad height");
DEFINE_int32(pad_w, 0, "pad width");
DEFINE_int32(stride_h, 2, "stride height");
DEFINE_int32(stride_w, 2, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_bool(flag_relu, false, "do relu");
DEFINE_bool(flag_bias, false, "with bias");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ConvParam ConvParam;
typedef paddle::lite::operators::ActivationParam ActivationParam;
using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::ConvParam& param) {
  auto filter_dims = param.filter->dims();
  DDim output_shape = dim_in;
  output_shape[1] = filter_dims[1] * param.groups;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  for (int i = 0; i < 2; i++) {
    int kernel_extent = dilations[i] * (filter_dims[i + 2] - 1) + 1;
    int output_len = (dim_in[i + 2] - 1) * param.strides[i] + kernel_extent -
                     (paddings[2 * i] + paddings[2 * i + 1]);
    output_shape[i + 2] = output_len;
  }
  return output_shape;
}

#ifdef LITE_WITH_ARM
void test_conv_transpose_fp32(const std::vector<DDim>& input_dims,
                              const DDim& weight_dim,
                              int group,
                              const std::vector<int>& strides,
                              const std::vector<int>& pads,
                              const std::vector<int>& dilas,
                              bool flag_bias,
                              bool flag_relu,
                              const std::vector<int>& thread_num,
                              const std::vector<int>& power_mode) {
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
    param.bias->Resize({weight_dim[1] * group});
    param.bias->set_precision(PRECISION(kFloat));
  }
  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.fuse_relu = flag_relu;
  param.groups = group;

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFloat));

  paddle::lite::fill_tensor_rand(*param.filter, -1.f, 1.f);
  // paddle::lite::fill_tensor_const(*param.filter, 1.f);
  if (flag_bias) {
    paddle::lite::fill_tensor_rand(*param.bias, -1.f, 1.f);
    // paddle::lite::fill_tensor_const(*param.bias, 1.f);
  }
  if (flag_relu) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type =
        (paddle::lite_api::ActivationType)1;  // 2-relu6 4-leakyrelu
    param.activation_param = act_param;
  }
  Tensor tmp_weights;
  tmp_weights.Resize(weight_dim);
  tmp_weights.CopyDataFrom(*param.filter);
  auto wptr = tmp_weights.data<float>();
  auto bias_ptr = flag_bias ? param.bias->data<float>() : nullptr;

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::Conv2DTransposeCompute<PRECISION(kFloat),
                                                         PRECISION(kFloat)>
          conv_t;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);

      for (auto& dim_in : input_dims) {
        param.x->Resize(dim_in);
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param.output->Resize(dim_out);
        break;
      }

      conv_t.SetParam(param);
      conv_t.SetContext(std::move(ctx1));
      // prepare for run
      conv_t.PrepareForRun();

      for (auto& dim_in : input_dims) {
        CHECK_EQ(weight_dim[0], dim_in[1])
            << "input channel must equal to weights channel";
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param.x->Resize(dim_in);
        param.output->Resize(dim_out);
        param.filter->CopyDataFrom(tmp_weights);
        paddle::lite::fill_tensor_rand(*param.x, -1.f, 1.f);
        // paddle::lite::fill_tensor_const(*param.x, 1.f);
        auto din = param.x->data<float>();

        Tensor tout_basic;
        if (FLAGS_check_result) {
          tout_basic.set_precision(PRECISION(kFloat));
          tout_basic.Resize(dim_out);
          fill_tensor_const(tout_basic, 0.f);
          auto dout_basic = tout_basic.mutable_data<float>();

          deconv_basic<float, float>(din,
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
                                     pads[3],
                                     pads[0],
                                     pads[1],
                                     flag_bias,
                                     flag_relu);
        }
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          conv_t.Launch();
        }
        /// compute
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          conv_t.Launch();
          t0.Stop();
        }

        float gops =
            2.f * tmp_weights.numel() * dim_in[0] * dim_in[2] * dim_in[3];
        LOG(INFO) << "conv fp32: input shape: " << dim_in << ", output shape"
                  << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
                  << ", min time: " << t0.LapTimes().Min()
                  << ", total GOPS: " << 1e-9 * gops
                  << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                  << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
          LOG(INFO) << "compare result, max diff: " << max_diff
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
                         << ", bias: " << (flag_bias ? "true" : "false")
                         << ", relu: " << (flag_relu ? "true" : "false")
                         << ", threads: " << th << ", power_mode: " << cls
                         << " failed!!\n";
            }
          }
        }
        LOG(INFO) << "test fp32 conv: input: " << dim_in
                  << ", output: " << dim_out << ", weight dim: " << weight_dim
                  << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
                  << ", " << pads[3] << ", stride: " << strides[0] << ", "
                  << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
                  << ", bias: " << (flag_bias ? "true" : "false")
                  << ", relu: " << (flag_relu ? "true" : "false")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.filter;
  delete param.output;
  delete param.bias;
}
#ifdef ENABLE_ARM_FP16
void test_conv_transpose_fp16(const std::vector<DDim>& input_dims,
                              const DDim& weight_dim,
                              int group,
                              const std::vector<int>& strides,
                              const std::vector<int>& pads,
                              const std::vector<int>& dilas,
                              bool flag_bias,
                              bool flag_relu,
                              const std::vector<int>& thread_num,
                              const std::vector<int>& power_mode) {
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
    param.bias->Resize({weight_dim[1] * group});
    param.bias->set_precision(PRECISION(kFP16));
  }
  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.fuse_relu = flag_relu;
  param.groups = group;

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFP16));

  auto weight_fp16 = param.filter->mutable_data<float16_t>();

  // fill_data_rand<float16_t>(weight_fp16, -1.f, 1.f, param.filter->numel());
  fill_data_const<float16_t>(weight_fp16, 1.f, param.filter->numel());
  if (flag_bias) {
    auto bias_fp16 = param.bias->mutable_data<float16_t>();
    // fill_data_rand<float16_t>(bias_fp16, -1.f, 1.f, param.filter->numel());
    fill_data_const<float16_t>(bias_fp16, 1.f, param.bias->numel());
  }
  if (flag_relu) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type =
        (paddle::lite_api::ActivationType)1;  // 2-relu6 4-leakyrelu
    param.activation_param = act_param;
  }
  Tensor tmp_weights;
  tmp_weights.Resize(weight_dim);
  tmp_weights.CopyDataFrom(*param.filter);
  auto wptr = tmp_weights.data<float16_t>();
  auto bias_ptr = flag_bias ? param.bias->data<float16_t>() : nullptr;

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::Conv2DTransposeCompute<PRECISION(kFP16),
                                                         PRECISION(kFP16)>
          conv_t;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      conv_t.SetParam(param);
      conv_t.SetContext(std::move(ctx1));
      for (auto& dim_in : input_dims) {
        CHECK_EQ(weight_dim[0], dim_in[1])
            << "input channel must equal to weights channel";
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param.x->Resize(dim_in);
        param.output->Resize(dim_out);
        param.filter->CopyDataFrom(tmp_weights);
        // prepare for run
        conv_t.PrepareForRun();
        auto din_fp16 = param.x->mutable_data<float16_t>();
        // fill_data_rand<float16_t>(bias_fp16, -1.f, 1.f, param.x->numel());
        fill_data_const<float16_t>(din_fp16, 1.f, param.x->numel());

        Tensor tout_basic;
        if (FLAGS_check_result) {
          tout_basic.set_precision(PRECISION(kFP16));
          tout_basic.Resize(dim_out);
          auto dout_basic_fp16 = tout_basic.mutable_data<float16_t>();
          fill_data_const<float16_t>(
              dout_basic_fp16, 1.f, dim_out.production());
          deconv_basic<float16_t, float16_t>(din_fp16,
                                             dout_basic_fp16,
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
                                             pads[3],
                                             pads[0],
                                             pads[1],
                                             flag_bias,
                                             flag_relu);
        }
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          conv_t.Launch();
        }
        /// compute
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          conv_t.Launch();
          t0.Stop();
        }

        float gops =
            2.f * tmp_weights.numel() * dim_in[0] * dim_in[2] * dim_in[3];
        LOG(INFO) << "conv fp16: input shape: " << dim_in << ", output shape"
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
          Tensor tdiff;
          tdiff.Resize(tout_basic.dims());
          tdiff.set_precision(PRECISION(kFP16));
          auto ptr = tdiff.mutable_data<float16_t>();
          data_diff(basic_ptr,
                    saber_ptr,
                    ptr,
                    tout_basic.numel(),
                    max_ratio,
                    max_diff);
          print_diff_info(max_diff, max_ratio);
          if (std::abs(max_ratio) > 1e-3f) {
            if (max_diff > 1e-1f) {
              int64_t size = tout_basic.numel();
              int64_t width = tout_basic.dims()[3];
              print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
              LOG(FATAL) << "test fp16 conv: input: " << dim_in
                         << ", output: " << dim_out
                         << ", weight dim: " << weight_dim
                         << ", pad: " << pads[0] << ", " << pads[1] << ", "
                         << pads[2] << ", " << pads[3]
                         << ", stride: " << strides[0] << ", " << strides[1]
                         << ", dila_: " << dilas[0] << ", " << dilas[1]
                         << ", bias: " << (flag_bias ? "true" : "false")
                         << ", relu: " << (flag_relu ? "true" : "false")
                         << ", threads: " << th << ", power_mode: " << cls
                         << " failed!!\n";
            }
          }
        }
        LOG(INFO) << "test fp16 conv: input: " << dim_in
                  << ", output: " << dim_out << ", weight dim: " << weight_dim
                  << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
                  << ", " << pads[3] << ", stride: " << strides[0] << ", "
                  << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
                  << ", bias: " << (flag_bias ? "true" : "false")
                  << ", relu: " << (flag_relu ? "true" : "false")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.filter;
  delete param.output;
  delete param.bias;
}
#endif
#else
void test_conv_transpose_fp32(const std::vector<DDim>& input_dims,
                              const DDim& weight_dim,
                              int group,
                              const std::vector<int>& strides,
                              const std::vector<int>& pads,
                              const std::vector<int>& dilas,
                              bool flag_bias,
                              bool flag_relu,
                              const std::vector<int>& thread_num,
                              const std::vector<int>& power_mode) {}

void test_conv_transpose_fp16(const std::vector<DDim>& input_dims,
                              const DDim& weight_dim,
                              int group,
                              const std::vector<int>& strides,
                              const std::vector<int>& pads,
                              const std::vector<int>& dilas,
                              bool flag_bias,
                              bool flag_relu,
                              const std::vector<int>& thread_num,
                              const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param conv
TEST(TestConvRand, test_conv_transpose_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 16}) {
      for (auto& cout : {1, 5, 8, 16}) {
        for (auto& g : {1, 2}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad_h0 : {0, 1, 2}) {
                  for (auto& pad_h1 : {0, 1, 2}) {
                    for (auto& pad_w0 : {0, 1, 2}) {
                      for (auto& pad_w1 : {0, 1, 2}) {
                        for (auto& dila : {1, 2}) {
                          for (auto& flag_bias : {false, true}) {
                            for (auto& flag_relu : {false, true}) {
                              if (cin % g != 0 || cout % g != 0 ||
                                  pad_h1 != pad_h0 || pad_w0 != pad_w1) {
                                continue;
                              }
                              std::vector<DDim> dims;
                              DDim weights_dim({cin, cout / g, kh, kw});
                              for (auto& batch : {2}) {
                                for (auto& h : {1, 3, 19, 32, 28}) {
                                  dims.push_back(DDim({batch, cin, h, h}));
                                }
                              }
                              test_conv_transpose_fp32(
                                  dims,
                                  weights_dim,
                                  g,
                                  {stride, stride},
                                  {pad_h0, pad_h1, pad_w0, pad_w1},
                                  {dila, dila},
                                  flag_bias,
                                  flag_relu,
                                  {4},
                                  {FLAGS_power_mode});
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
#ifdef ENABLE_ARM_FP16
TEST(TestConvRand, test_conv_transpose_fp16_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {1, 5, 8}) {
        for (auto& g : {1, 2}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad_h0 : {0, 1, 2}) {
                  for (auto& pad_h1 : {0, 1, 2}) {
                    for (auto& pad_w0 : {0, 1, 2}) {
                      for (auto& pad_w1 : {0, 1, 2}) {
                        for (auto& dila : {1, 2}) {
                          for (auto& flag_bias : {false, true}) {
                            for (auto& flag_relu : {false, true}) {
                              if (cin % g != 0 || cout % g != 0) {
                                continue;
                              }
                              std::vector<DDim> dims;
                              DDim weights_dim({cin, cout / g, kh, kw});
                              for (auto& batch : {2}) {
                                for (auto& h : {1, 3, 19, 32, 28}) {
                                  dims.push_back(DDim({batch, cin, h, h}));
                                }
                              }
                              test_conv_transpose_fp16(
                                  dims,
                                  weights_dim,
                                  g,
                                  {stride, stride},
                                  {pad_h0, pad_h1, pad_w0, pad_w1},
                                  {dila, dila},
                                  flag_bias,
                                  flag_relu,
                                  {4},
                                  {FLAGS_power_mode});
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
TEST(TestConvCustom, test_conv_transpose_fp16_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  test_conv_transpose_fp16(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      DDim({FLAGS_in_channel,
            FLAGS_out_channel / FLAGS_group,
            FLAGS_kernel_h,
            FLAGS_kernel_w}),
      FLAGS_group,
      {FLAGS_stride_h, FLAGS_stride_w},
      {FLAGS_pad_h, FLAGS_pad_h, FLAGS_pad_w, FLAGS_pad_w},
      {FLAGS_dila_h, FLAGS_dila_w},
      FLAGS_flag_bias,
      FLAGS_flag_relu,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif
#if 1  /// custom
TEST(TestConvCustom, test_conv_transpose_fp32_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  test_conv_transpose_fp32(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      DDim({FLAGS_in_channel,
            FLAGS_out_channel / FLAGS_group,
            FLAGS_kernel_h,
            FLAGS_kernel_w}),
      FLAGS_group,
      {FLAGS_stride_h, FLAGS_stride_w},
      {FLAGS_pad_h, FLAGS_pad_h, FLAGS_pad_w, FLAGS_pad_w},
      {FLAGS_dila_h, FLAGS_dila_w},
      FLAGS_flag_bias,
      FLAGS_flag_relu,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif  // custom
