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

#include "lite/kernels/arm/conv_compute.h"
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/core/context.h"
#include "lite/tests/utils/tensor_utils.h"
#include "lite/tests/utils/test_funcs.h"
#include "lite/tests/utils/test_lite.h"
#include "lite/tests/utils/timer.h"

typedef lite::test::TestLite TestLite;

int g_cluster = 0;
int g_threads = 1;
int g_warmup_iter = 0;
int g_test_iter = 1;
bool g_basic_test = true;
bool g_compare_result = true;

int g_num = 1;
int g_ch_in = 32;
int g_h_in = 112;
int g_w_in = 112;

int g_ch_out = 32;
int g_group = 1;
int g_kw = 3;
int g_pad_w = 1;
int g_stride_w = 1;
int g_dila_w = 1;
int g_kh = 3;
int g_pad_h = 1;
int g_stride_h = 1;
int g_dila_h = 1;

bool g_flag_relu = true;
bool g_flag_bias = true;

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

bool test_conv_fp32(int n,
                    int c,
                    int h,
                    int w,
                    int num_out,
                    int group,
                    const std::vector<int>& kernels,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    bool flag_relu,
                    int thread_num,
                    int cluster_id) {
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
#ifdef LITE_WITH_ARM
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cluster_id),
                 thread_num);
#else
  return true;
#endif
  paddle::lite::DDim dim_in{{n, c, h, w}};
  DDim dim_w{{num_out, c / group, kernels[0], kernels[1]}};
  ConvParam param;
  param.x = new Tensor;
  param.x->Resize(dim_in);
  param.x->set_precision(PRECISION(kFloat));
  param.filter = new Tensor;
  param.filter->Resize(dim_w);
  param.filter->set_precision(PRECISION(kFloat));
  if (flag_bias) {
    param.bias = new Tensor;
    param.bias->Resize({num_out});
    param.bias->set_precision(PRECISION(kFloat));
  }
  param.strides = strides;
  param.paddings = pads;
  param.dilations = dilas;
  param.fuse_relu = flag_relu;
  param.groups = group;

  DDim dim_out = compute_out_dim(dim_in, param);
  if (dim_out[2] < 1 || dim_out[3] < 1) {
    return true;
  }
  param.output = new Tensor;
  param.output->Resize(dim_out);
  param.output->set_precision(PRECISION(kFloat));

  Tensor tout_basic;

  //  paddle::lite::fill_tensor_rand(*param.x, -1.f, 1.f);
  //  paddle::lite::fill_tensor_rand(*param.filter, -1.f, 1.f);
  paddle::lite::fill_tensor_const(*param.x, 1.f);
  paddle::lite::fill_tensor_const(*param.filter, 1.f);
  if (flag_bias) {
    //    paddle::lite::fill_tensor_rand(*param.bias, -1.f, 1.f);
    paddle::lite::fill_tensor_const(*param.bias, 1.f);
  }

  auto din = param.x->data<float>();
  auto wptr = param.filter->data<float>();
  auto bias_ptr = flag_bias ? param.bias->data<float>() : nullptr;

  if (g_compare_result) {
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
                             dim_w[3],
                             dim_w[2],
                             strides[1],
                             strides[0],
                             dilas[1],
                             dilas[0],
                             pads[1],
                             pads[0],
                             flag_bias,
                             flag_relu);
  }

  auto conv = new paddle::lite::kernels::arm::ConvCompute<PRECISION(kFloat),
                                                          PRECISION(kFloat)>;
  conv->SetContext(std::move(ctx1));
  conv->SetParam(param);
  /// prepare for run
  conv->PrepareForRun();
  /// warm up
  for (int i = 0; i < g_warmup_iter; ++i) {
    conv->Launch();
  }
  /// compute
  lite::test::Timer t0;
  for (int i = 0; i < g_test_iter; ++i) {
    t0.start();
    conv->Launch();
    t0.end();
  }

  double gops = 2.0 * dim_out.production() * dim_in[1] * dim_w[2] * dim_w[3] /
                param.groups;
  LOG(INFO) << "conv fp32: output shape" << dim_out
            << ",running time, avg: " << t0.get_average_ms()
            << ", min time: " << t0.get_min_time()
            << ", total GOPS: " << 1e-9 * gops
            << " GOPS, avg GOPs: " << 1e-6 * gops / t0.get_average_ms()
            << " GOPs, max GOPs: " << 1e-6 * gops / t0.get_min_time();

  bool res = true;
  if (g_compare_result) {
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
        res = false;
      }
    }
  }
  delete param.x;
  delete param.filter;
  delete param.output;
  if (param.bias) {
    delete param.bias;
  }
  delete conv;
  return res;
}

#if 1  /// 3x3dw
TEST(TestLite, test_conv_depthwise) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
      for (auto& c : {1, 3, 5, 8, 16, 32}) {
        for (auto& h : {1, 3, 8, 15, 19, 32, 38, 56, 75}) {
          for (auto& stride : {1}) {
            for (auto& flag_bias : {false, true}) {
              for (auto& flag_relu : {false, true}) {
                //！ fix me, pad == 0 no pass
                for (auto& pad : {0, 1}) {
                  for (auto& th : {1, 2, 4}) {
                    int w = h;
                    if (!test_conv_fp32(batch,
                                        c,
                                        h,
                                        w,
                                        c,
                                        c,
                                        {3, 3},
                                        {stride, stride},
                                        {pad, pad},
                                        {1, 1},
                                        flag_bias,
                                        flag_relu,
                                        th,
                                        g_cluster)) {
                      LOG(FATAL)
                          << "test fp32 3x3 depthwise conv: batchsize: "
                          << batch << ", channel: " << c << ", h & w: " << h
                          << ", stride: " << stride << ", pad: " << pad
                          << ", bias: " << (flag_bias ? "true" : "false")
                          << ", relu: " << (flag_relu ? "true" : "false")
                          << ", threads: " << th << ", cluster: " << g_cluster
                          << " failed!!\n";
                      return;
                    }
                    LOG(INFO)
                        << "test fp32 3x3 depthwise conv: batchsize: " << batch
                        << ", channel: " << c << ", h & w: " << h
                        << ", stride: " << stride << ", pad: " << pad
                        << ", bias: " << (flag_bias ? "true" : "false")
                        << ", relu: " << (flag_relu ? "true" : "false")
                        << ", threads: " << th << ", cluster: " << g_cluster
                        << " passed!!\n";
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
#endif  // 3x3dw

#if 0   /// 5x5dw
TEST(TestLite, test_conv_depthwise) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto& c : {1, 3, 5, 8}) {
    for (auto& h : {1, 3, 8, 15}) {
    for (auto& stride : {1, 2}) {
    for (auto& flag_bias : {true}) {
    for (auto& flag_relu : {true}) {
    //！ fix me, pad == 0 no pass
    for (auto& pad : {0, 1}) {
    for (auto& th : {1, 2, 4}) {
      int w = h;
      if (h == 1 && stride == 2) {
        continue;
      }
      if (!test_conv_fp32(batch,
                          c,
                          h,
                          w,
                          c,
                          c,
                          {3, 3},
                          {stride, stride},
                          {pad, pad},
                          {1, 1},
                          flag_bias,
                          flag_relu,
                          th,
                          g_cluster)) {
        LOG(FATAL) << "test fp32 5x5 depthwise conv: batchsize: "
                   << batch << ", channel: " << c << ", h & w: " << h
                   << ", stride: " << stride << ", pad: " << pad << ", bias: "
                   << (flag_bias? "true" : "false") << ", relu: "
                   << (flag_relu? "true" : "false") << ", threads: "
                   << th << ", cluster: " << g_cluster << " failed!!\n";
        return;
      }
      LOG(INFO) << "test fp32 5x5 depthwise conv: batchsize: "
                << batch << ", channel: " << c << ", h & w: " << h
                << ", stride: " << stride << ", pad: " << pad << ", bias: "
                << (flag_bias? "true" : "false") << ", relu: "
                << (flag_relu? "true" : "false") << ", threads: "
                << th << ", cluster: " << g_cluster << " passed!!\n";
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
#endif  // 5x5dw

#if 0   /// conv1x1s1
TEST(TestLite, test_conv_1x1s1) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
      for (auto& c : {1, 3, 8, 11, 32, 128}) {
        for (auto& cout : {1, 5, 16, 37, 128}) {
          for (auto& g_div : {1, 2}) {
            for (auto& h : {1, 7, 31, 56, 75, 100, 240}) {
              for (auto& flag_bias : {false, true}) {
                for (auto& flag_relu : {false, true}) {
                  for (auto& th : {1, 2, 4}) {
                    int w = h;
                    int g = g_div;
                    if (c % g != 0 || cout % g != 0) {
                      continue;
                    }
                    if (!test_conv_fp32(batch,
                                        c,
                                        h,
                                        w,
                                        cout,
                                        g,
                                        {1, 1},
                                        {1, 1},
                                        {0, 0},
                                        {1, 1},
                                        flag_bias,
                                        flag_relu,
                                        th,
                                        g_cluster)) {
                      LOG(ERROR) << "test fp32 1x1 conv: batchsize: " << batch
                                 << ", channel: " << c << ", h & w: " << h
                                 << ", bias: " << (flag_bias ? "true" : "false")
                                 << ", relu: " << (flag_relu ? "true" : "false")
                                 << ", threads: " << th
                                 << ", cluster: " << g_cluster << " failed!!\n";
                      return;
                    }
                    LOG(INFO) << "test fp32 1x1 conv: batchsize: " << batch
                              << ", channel: " << c << ", h & w: " << h
                              << ", bias: " << (flag_bias ? "true" : "false")
                              << ", relu: " << (flag_relu ? "true" : "false")
                              << ", threads: " << th
                              << ", cluster: " << g_cluster << " passed!!\n";
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
#endif  // conv1x1s1

#if 0   /// conv3x3s1
TEST(TestLite, test_conv_3x3s1) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto& cin : {1, 3, 8, 11, 32}) {
    for (auto& cout : {1, 5, 8, 37, 64}) {
    for (auto& h : {1, 3, 31, 56, 75, 100}) {
    for (auto& pad : {1, 2}) {
    for (auto& flag_bias : {false, true}) {
    for (auto& flag_relu : {false, true}) {
    for (auto& th : {1, 2, 4}) {
      int w = h;
      if (!test_conv_fp32(batch,
                          cin,
                          h,
                          w,
                          cout,
                          1,
                          {3, 3},
                          {1, 1},
                          {pad, pad},
                          {1, 1},
                          flag_bias,
                          flag_relu,
                          th,
                          g_cluster)) {
        LOG(FATAL) << "test fp32 3x3s1 conv: batchsize: " << batch
                   << ", channel: " << cin << ", h & w: " << h
                   << ", num_out: " << cout << ", pad: " << pad
                   << ", bias: " << (flag_bias ? "true" : "false")
                   << ", relu: " << (flag_relu ? "true" : "false")
                   << ", threads: " << th
                   << ", cluster: " << g_cluster << " failed!!\n";
        return;
      }
      LOG(INFO) << "test fp32 3x3s1 conv: batchsize: " << batch
                << ", channel: " << cin << ", h & w: " << h
                << ", num_out: " << cout << ", pad: " << pad
                << ", bias: " << (flag_bias ? "true" : "false")
                << ", relu: " << (flag_relu ? "true" : "false")
                << ", threads: " << th
                << ", cluster: " << g_cluster << " passed!!\n";
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
#endif  // conv3x3s1

#if 0   /// conv3x3s2
TEST(TestLite, test_conv_3x3s2) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto &cin : {1, 3, 8, 11, 32}) {
    for (auto& cout : {1, 8, 15, 32, 64}) {
    for (auto &h : {3, 19, 32, 56, 75, 100}) {
    for (auto &pad : {1, 2}) {
    for (auto &flag_bias : {false, true}) {
    for (auto &flag_relu : {false, true}) {
    for (auto &th : {1, 2, 4}) {
      int w = h;
      if (!test_conv_fp32(batch,
                          cin,
                          h,
                          w,
                          cout,
                          1,
                          {3, 3},
                          {2, 2},
                          {pad, pad},
                          {1, 1},
                          flag_bias,
                          flag_relu,
                          th,
                          g_cluster)) {
        LOG(FATAL) << "test fp32 3x3s2 conv: batchsize: "
                   << batch << ", channel: " << cin
                   << ", h & w: " << h << ", num_out: " << cout
                   << ", bias: " << (flag_bias ? "true" : "false")
                   << ", relu: " << (flag_relu ? "true" : "false")
                   << ", threads: " <<  th
                   << ", cluster: " << g_cluster << " failed!!\n";
        return;
      }
      LOG(INFO) << "test fp32 3x3s2 conv: batchsize: "
                << batch << ", channel: " << cin
                << ", h & w: " << h << ", num_out: " << cout
                << ", bias: " << (flag_bias ? "true" : "false")
                << ", relu: " << (flag_relu ? "true" : "false")
                << ", threads: " << th
                << ", cluster: " << g_cluster << " passed!!\n";
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
#endif  // conv3x3s2

#if 1  /// custom
TEST(TestLite, test_conv_fp32_custom_size) {
  auto flag = test_conv_fp32(g_num,
                             g_ch_in,
                             g_h_in,
                             g_w_in,
                             g_ch_out,
                             g_group,
                             {g_kh, g_kw},
                             {g_stride_h, g_stride_w},
                             {g_pad_h, g_pad_w},
                             {g_dila_h, g_dila_w},
                             g_flag_bias,
                             g_flag_relu,
                             g_threads,
                             g_cluster);
  if (flag) {
    LOG(INFO) << "test fp32 conv: batchsize: " << g_num
              << ", channel: " << g_ch_in << ", h: " << g_h_in
              << ", w: " << g_w_in << ", num_out: " << g_ch_out
              << ", group: " << g_group << ", kw: " << g_kw << ", kh: " << g_kh
              << ", pad_w: " << g_pad_w << ", pad_h: " << g_pad_h
              << ", stride_w: " << g_stride_w << ", stride_h: " << g_stride_h
              << ", dila_w: " << g_dila_w << ", dila_h: " << g_dila_h
              << ", bias: " << (g_flag_bias ? "true" : "false")
              << ", relu: " << (g_flag_relu ? "true" : "false")
              << ", threads: " << g_threads << ", cluster: " << g_cluster
              << " passed!!\n";
  } else {
    LOG(FATAL) << "test fp32 conv: batchsize: " << g_num
               << ", channel: " << g_ch_in << ", h: " << g_h_in
               << ", w: " << g_w_in << ", num_out: " << g_ch_out
               << ", group: " << g_group << ", kw: " << g_kw << ", kh: " << g_kh
               << ", pad_w: " << g_pad_w << ", pad_h: " << g_pad_h
               << ", stride_w: " << g_stride_w << ", stride_h: " << g_stride_h
               << ", dila_w: " << g_dila_w << ", dila_h: " << g_dila_h
               << ", bias: " << (g_flag_bias ? "true" : "false")
               << ", relu: " << (g_flag_relu ? "true" : "false")
               << ", threads: " << g_threads << ", cluster: " << g_cluster
               << " failed!!\n";
  }
}
#endif  // custom

int main(int argc, const char** argv) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  LOG(INFO)
      << "usage: ./" << argv[0]
      << " basic_test cluster  threads  warmup test_iter "
      << " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group"
      << " kernel pad stride dila [kernel_h] [pad_h] [stride_h] [dila_h]";

  if (argc >= 2) {
    g_basic_test = atoi(argv[1]) > 0;
  }

  if (argc >= 3) {
    g_cluster = atoi(argv[2]);
  }
  if (argc >= 4) {
    g_threads = atoi(argv[3]);
  }
  if (argc >= 5) {
    g_test_iter = atoi(argv[4]);
  }
  if (argc >= 6) {
    g_test_iter = atoi(argv[5]);
  }
  if (argc >= 7) {
    g_compare_result = atoi(argv[6]) > 0;
  }
  if (argc >= 8) {
    g_flag_bias = atoi(argv[7]) > 0;
  }
  if (argc >= 9) {
    g_flag_relu = atoi(argv[8]) > 0;
  }
  if (argc >= 10) {
    if (argc < 19) {
      LOG(FATAL)
          << "usage: ./" << argv[0]
          << " basic_test cluster  threads warmup test_iter "
          << " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out "
             "group"
          << " kernel pad stride dila [kernel_h] [pad_h] [stride_h] [dila_h]";
      return -1;
    }
    g_num = atoi(argv[9]);
    g_ch_in = atoi(argv[10]);
    g_h_in = atoi(argv[11]);
    g_w_in = atoi(argv[12]);
    g_ch_out = atoi(argv[13]);
    g_group = atoi(argv[14]);
    g_kw = atoi(argv[15]);
    g_kh = g_kw;
    g_pad_w = atoi(argv[16]);
    g_pad_h = g_pad_w;
    g_stride_w = atoi(argv[17]);
    g_stride_h = g_stride_w;
    g_dila_w = atoi(argv[18]);
    g_dila_h = g_dila_w;
  }
  if (argc > 19) {
    g_kh = atoi(argv[19]);
  }
  if (argc > 20) {
    g_pad_h = atoi(argv[20]);
  }
  if (argc > 21) {
    g_stride_h = atoi(argv[21]);
  }
  if (argc > 22) {
    g_dila_h = atoi(argv[22]);
  }

  // initial logger
  // logger::init(argv[0]);
  InitTest();
  RUN_ALL_TESTS(argv[0]);
  return 0;
}
