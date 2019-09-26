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
#include "lite/operators/op_params.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/tensor_utils.h"
#include "lite/tests/utils/timer.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/conv_compute.h"
#endif  // LITE_WITH_ARM

DEFINE_int32(cluster, 3, "cluster id");
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
DEFINE_int32(group, 1, "group");
DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h, 1, "pad height");
DEFINE_int32(pad_w, 1, "pad width");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_bool(flag_relu, true, "do relu");
DEFINE_bool(flag_bias, true, "with bias");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ConvParam ConvParam;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::ConvParam& param) {
  DDim dim_out = dim_in;
  dim_out[1] = param.filter->dims()[0];
  auto kernel_h = param.filter->dims()[2];
  auto kernel_w = param.filter->dims()[3];
  auto h = dim_in[2];
  auto w = dim_in[3];
  int dila_h = param.dilations[0];
  int dila_w = param.dilations[1];
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  auto hout = (h + 2 * pad_h - kernel_exten) / stride_h + 1;
  kernel_exten = dila_w * (kernel_w - 1) + 1;
  auto wout = (w + 2 * pad_w - kernel_exten) / stride_w + 1;
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
                    bool flag_relu,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& cluster_id) {
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
  param.paddings = pads;
  param.dilations = dilas;
  param.fuse_relu = flag_relu;
  param.groups = group;

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

  for (auto& cls : cluster_id) {
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
        //        paddle::lite::fill_tensor_const(*param.x, 1.f);
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
                                   pads[1],
                                   pads[0],
                                   flag_bias,
                                   flag_relu);
        }
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          conv.Launch();
        }
        /// compute
        lite::test::Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.start();
          conv.Launch();
          t0.end();
        }

        double gops = 2.0 * dim_out.production() * dim_in[1] * weight_dim[2] *
                      weight_dim[3] / param.groups;
        LOG(INFO) << "conv fp32: input shape: " << dim_in << ", output shape"
                  << dim_out << ",running time, avg: " << t0.get_average_ms()
                  << ", min time: " << t0.get_min_time()
                  << ", total GOPS: " << 1e-9 * gops
                  << " GOPS, avg GOPs: " << 1e-6 * gops / t0.get_average_ms()
                  << " GOPs, max GOPs: " << 1e-6 * gops / t0.get_min_time();

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
              LOG(WARNING) << "saber result";
              print_tensor(*param.output);
              Tensor tdiff;
              tdiff.Resize(tout_basic.dims());
              tdiff.set_precision(PRECISION(kFloat));
              tensor_diff(tout_basic, *param.output, tdiff);
              print_tensor(tdiff);
              LOG(FATAL) << "test fp32 conv: input: " << dim_in
                         << ", output: " << dim_out
                         << ", weight dim: " << weight_dim
                         << ", pad: " << pads[0] << ", " << pads[1]
                         << ", stride: " << strides[0] << ", " << strides[1]
                         << ", dila_: " << dilas[0] << ", " << dilas[1]
                         << ", bias: " << (flag_bias ? "true" : "false")
                         << ", relu: " << (flag_relu ? "true" : "false")
                         << ", threads: " << th << ", cluster: " << cls
                         << " failed!!\n";
            }
          }
        }
        LOG(INFO) << "test fp32 conv: input: " << dim_in
                  << ", output: " << dim_out << ", weight dim: " << weight_dim
                  << ", pad: " << pads[0] << ", " << pads[1]
                  << ", stride: " << strides[0] << ", " << strides[1]
                  << ", dila_: " << dilas[0] << ", " << dilas[1]
                  << ", bias: " << (flag_bias ? "true" : "false")
                  << ", relu: " << (flag_relu ? "true" : "false")
                  << ", threads: " << th << ", cluster: " << cls
                  << " successed!!\n";
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
                    bool flag_relu,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& cluster_id) {}
#endif  // LITE_WITH_ARM

#if 1  /// 3x3dw
TEST(TestConv3x3DW, test_conv3x3_depthwise) {
  if (FLAGS_basic_test) {
    for (auto& stride : {1, 2}) {
      for (auto& pad : {0, 1}) {
        for (auto& flag_bias : {false, true}) {
          for (auto& flag_relu : {false, true}) {
            for (auto& c : {1, 3, 5, 8, 16, 32}) {
              std::vector<DDim> dims;
              DDim weights_dim({c, 1, 3, 3});
              for (auto& batch : {1, 2}) {
                for (auto& h : {1, 3, 15, 19, 28, 32, 75}) {
                  dims.push_back(DDim({batch, c, h, h}));
                }
              }
              test_conv_fp32(dims,
                             weights_dim,
                             c,
                             {stride, stride},
                             {pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             {1, 2, 4},
                             {FLAGS_cluster});
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
      for (auto& pad : {0, 1, 2}) {
        for (auto& flag_bias : {false, true}) {
          for (auto& flag_relu : {false, true}) {
            for (auto& c : {1, 3, 5, 8, 16, 32}) {
              std::vector<DDim> dims;
              DDim weights_dim({c, 1, 5, 5});
              for (auto& batch : {1, 2}) {
                for (auto& h : {1, 3, 15, 19, 28, 32, 75}) {
                  dims.push_back(DDim({batch, c, h, h}));
                }
              }
              test_conv_fp32(dims,
                             weights_dim,
                             c,
                             {stride, stride},
                             {pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             {1, 2, 4},
                             {FLAGS_cluster});
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
            for (auto& flag_relu : {false, true}) {
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
              test_conv_fp32(dims,
                             weights_dim,
                             g,
                             {1, 1},
                             {0, 0},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             {1, 2, 4},
                             {FLAGS_cluster});
            }
          }
        }
      }
    }
  }
}
#endif  /// conv1x1s1

#if 1  /// conv3x3s1
TEST(TestConv3x3s1, test_conv_3x3s1) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 32, 48}) {
      for (auto& cout : {1, 5, 8, 32, 48}) {
        for (auto& pad : {1, 2}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_relu : {false, true}) {
              std::vector<DDim> dims;
              DDim weights_dim({cout, cin, 3, 3});
              for (auto& batch : {1, 2}) {
                for (auto& h : {1, 7, 19, 56, 32}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                }
              }
              test_conv_fp32(dims,
                             weights_dim,
                             1,
                             {1, 1},
                             {pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             {1, 2, 4},
                             {FLAGS_cluster});
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
    for (auto& cin : {1, 3, 8, 32}) {
      for (auto& cout : {1, 5, 8, 32}) {
        for (auto& pad : {1, 2}) {
          for (auto& flag_bias : {false, true}) {
            for (auto& flag_relu : {false, true}) {
              std::vector<DDim> dims;
              DDim weights_dim({cout, cin, 3, 3});
              for (auto& batch : {1, 2}) {
                for (auto& h : {1, 7, 19, 28, 75, 56, 32}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                }
              }
              test_conv_fp32(dims,
                             weights_dim,
                             1,
                             {2, 2},
                             {pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             {1, 2, 4},
                             {FLAGS_cluster});
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
    for (auto& cin : {1, 3, 8, 16}) {
      for (auto& cout : {1, 5, 8, 16}) {
        for (auto& g : {1, 2}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad : {0, 1, 2}) {
                  for (auto& dila : {1, 2}) {
                    for (auto& flag_bias : {false, true}) {
                      for (auto& flag_relu : {false, true}) {
                        if (cin % g != 0 || cout % g != 0) {
                          continue;
                        }
                        std::vector<DDim> dims;
                        DDim weights_dim({cout, cin / g, kh, kw});
                        for (auto& batch : {1, 2}) {
                          for (auto& h : {1, 3, 19, 32, 28}) {
                            dims.push_back(DDim({batch, cin, h, h}));
                          }
                        }
                        test_conv_fp32(dims,
                                       weights_dim,
                                       g,
                                       {stride, stride},
                                       {pad, pad},
                                       {dila, dila},
                                       flag_bias,
                                       flag_relu,
                                       {1, 2, 4},
                                       {FLAGS_cluster});
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
      {FLAGS_pad_h, FLAGS_pad_w},
      {FLAGS_dila_h, FLAGS_dila_w},
      FLAGS_flag_bias,
      FLAGS_flag_relu,
      {FLAGS_threads},
      {FLAGS_cluster});
}
#endif  // custom
