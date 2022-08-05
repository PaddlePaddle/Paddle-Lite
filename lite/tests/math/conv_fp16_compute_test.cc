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

#include "lite/tests/math/conv_ut.h"

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
  const float six = 6.f;
  act_init(
      param, strides, pads, dilas, group, flag_act, six, leakey_relu_scale);

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

      for (auto& dim_in : input_dims) {
        param.x->Resize(dim_in);
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          return;
        }
        param.output->Resize(dim_out);
        break;
      }

      conv.SetParam(param);
      conv.SetContext(std::move(ctx1));
      /// prepare for run
      conv.PrepareForRun();

      for (auto& dim_in : input_dims) {
        CHECK_EQ(weight_dim[1] * group, dim_in[1])
            << "input channel must equal to weights channel";
        param.x->Resize(dim_in);
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          return;
        }
        param.output->Resize(dim_out);

        Tensor x_fp32;
        x_fp32.Resize(dim_in);
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
                                   dim_in[0],
                                   dim_out[1],
                                   dim_out[2],
                                   dim_out[3],
                                   dim_in[1],
                                   dim_in[2],
                                   dim_in[3],
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
          print_diff_info(basic_max_diff, basic_max_ratio);
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
        print_gops_info("conv_fp16", dim_in, dim_out, t0, gops);
        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          auto basic_ptr = tout_basic_fp16.data<float16_t>();
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
          if ((max_diff - basic_max_diff) > 5e-4f) {
            int64_t size = tout_basic.numel();
            int count = 0;
            bool check = false;
            for (int i = 0; i < size; i++) {
              if (fabs(basic_ptr[i] - saber_ptr[i]) > 1e-1f &&
                  fabs(basic_ptr[i] - saber_ptr[i]) /
                          (fmax(fabs(basic_ptr[i]), fabs(saber_ptr[i]))) >
                      0.05) {
                check = true;
              }
            }
            if (check) {
              int64_t width = tout_basic.dims()[tout_basic.dims().size() - 1];
              print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
              print_conv_success_or_fail_info("conv_fp16",
                                              false,
                                              dim_in,
                                              dim_out,
                                              weight_dim,
                                              pads,
                                              strides,
                                              dilas,
                                              group,
                                              flag_bias,
                                              flag_act,
                                              th,
                                              cls);
            }
          }
        }
        print_conv_success_or_fail_info("conv_fp16",
                                        true,
                                        dim_in,
                                        dim_out,
                                        weight_dim,
                                        pads,
                                        strides,
                                        dilas,
                                        group,
                                        flag_bias,
                                        flag_act,
                                        th,
                                        cls);
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

#if 1  /// common dw
TEST(TestConvCommonDwFp16, test_conv_common_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride0 : {3}) {
      for (auto& stride1 : {2, 1}) {
        for (auto& dia0 : {2}) {
          for (auto& dia1 : {1, 2}) {
            for (auto& pad : {0}) {
              for (auto& pad1 : {1, 2}) {
                for (auto& flag_bias : {false, true}) {
                  for (auto& flag_act : {0, 1, 2, 4}) {
                    for (auto& c : {1, 5, 12, 32, 35, 46}) {
                      for (auto kh : {2, 3}) {
                        for (auto kw : {2, 3}) {
                          std::vector<DDim> dims;
                          for (auto& batch : {2}) {
                            for (auto& h : {8, 15, 25, 36, 46, 74, 108}) {
                              dims.push_back(DDim({batch, c, h, h}));
                            }
                          }
                          DDim weights_dim({c, 1, kh, kw});
                          const float leakey_relu_scale = 1.0f;
                          test_conv_fp16(dims,
                                         weights_dim,
                                         c,
                                         {stride0, stride1},
                                         {pad, pad1, pad1, pad1},
                                         {dia0, dia1},
                                         flag_bias,
                                         flag_act,
                                         {FLAGS_threads},
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
#endif  /// common dw

#if 1  /// 3x3dw
TEST(TestConv3x3DwFp16, test_conv3x3_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride : {1, 2}) {
      for (auto& pad : {0, 1}) {
        for (auto& pad1 : {0, 1}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_act : {0, 1, 2, 4}) {
              for (auto& c : {4, 7, 8, 10, 11, 16}) {
                DDim weights_dim({c, 1, 3, 3});
                std::vector<DDim> dims;

                for (auto& batch : {1}) {
                  for (auto& h : {33, 4, 18, 7, 17, 12, 16, 13, 15, 14}) {
                    dims.push_back(DDim({batch, c, h, h}));
                  }
                }
                const float leakey_relu_scale = 1.0f;
                test_conv_fp16(dims,
                               weights_dim,
                               c,
                               {stride, stride},
                               {pad, pad1, pad, pad1},
                               {1, 1},
                               flag_bias,
                               flag_act,
                               {FLAGS_threads},
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
                      for (auto& h : {56, 1, 15, 3, 56}) {
                        dims.push_back(DDim({batch, c, h, h}));
                      }
                    }
                    const float leakey_relu_scale = 1.0f;
                    test_conv_fp16(dims,
                                   weights_dim,
                                   c,
                                   {stride, stride},
                                   {pad_top, pad_bottom, pad_left, pad_right},
                                   {1, 1},
                                   flag_bias,
                                   flag_act,
                                   {FLAGS_threads},
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
                    std::vector<DDim> dims;
                    for (auto& batch : {1, 2}) {
                      for (auto& h : {32, 3, 56, 7, 15}) {
                        dims.push_back(DDim({batch, cin, h, h}));
                      }
                    }
                    const float leakey_relu_scale = 1.0f;
                    test_conv_fp16(dims,
                                   weights_dim,
                                   1,
                                   {2, 2},
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
#endif  /// conv3x3s2

#if 1  /// conv1x1s1
TEST(TestConv1x1s1, test_conv1x1s1) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 11, 32}) {
      for (auto& cout :
           {5, 16, 37}) {  // m=1 gemv_trans run one case is ok, but run more
                           // case in successive has diff.
        for (auto& g : {1, 2}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_act : {0, 1, 2, 4}) {
              std::vector<DDim> dims;
              if (cin % g != 0 || cout % g != 0) {
                continue;
              }
              DDim weights_dim({cout, cin / g, 1, 1});
              for (auto& batch : {1, 2}) {
                for (auto& h : {56, 1, 7, 32, 3, 28, 32, 1}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                }
              }
              const float leakey_relu_scale = 1.0f;
              test_conv_fp16(dims,
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
#endif  /// conv1x1s1

#if 1  /// random param conv
TEST(TestConvRand, test_conv_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {3, 5, 16}) {
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
                                for (auto& h : {1, 19, 3, 32}) {
                                  DDim dim_in({batch, cin, h, h});
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
                                  dims.push_back(DDim({batch, cin, h, h}));
                                }
                              }
                              if (dims.size() != 0) {
                                const float leakey_relu_scale = 1.0f;
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
}
#endif  /// random param conv

#if 1  /// custom
TEST(TestConvCustom, test_conv_fp16_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  std::vector<DDim> dims;
  dims.push_back(
      DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width}));
  test_conv_fp16(dims,
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

#if 1  /// conv3x3s2
TEST(TestDWConv3x3s2, test_dwconv_3x3s2) {
  if (FLAGS_basic_test) {
    for (auto& cin : {2, 6, 32}) {
      for (auto& cout : {cin}) {
        for (auto& pad : {0, 1}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_act : {0, 1, 2, 4}) {
              if (cin == 1 && cout == 1) {
                continue;
              }
              DDim weights_dim({cout, 1, 3, 3});
              std::vector<DDim> dims;
              for (auto& batch : {1, 4}) {
                for (auto& h : {224, 4, 112, 10, 100, 16, 96, 32, 48}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                }
              }
              const float leakey_relu_scale = 1.0f;
              test_conv_fp16(dims,
                             weights_dim,
                             cin,
                             {2, 2},
                             {pad, pad, pad, pad},
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
#endif  /// dwconv3x3s2
