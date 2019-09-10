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

#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/core/context.h"
#include "lite/kernels/arm/conv_compute.h"
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

template <paddle::lite::PrecisionType ptype>
void get_conv_param(int n,
                    int c,
                    int h,
                    int w,
                    int num_out,
                    int g,
                    const std::vector<int>& kernels,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dila,
                    bool flag_bias,
                    bool flag_relu,
                    ConvParam* param) {
  DDim dim_in({n, c, h, w});
  param->x = new Tensor;
  param->x->Resize(dim_in);
  param->x->set_precision(PRECISION(kInt8));
  param->filter = new Tensor;
  param->filter->Resize({num_out, c / g, kernels[0], kernels[1]});
  param->filter->set_precision(PRECISION(kInt8));
  if (flag_bias) {
    param->bias = new Tensor;
    param->bias->Resize({num_out});
    param->bias->set_precision(PRECISION(kFloat));
  }
  param->strides = strides;
  param->paddings = pads;
  param->dilations = dila;
  param->fuse_relu = flag_relu;
  param->groups = g;

  DDim dim_out = compute_out_dim(dim_in, *param);
  param->output = new Tensor;
  param->output->Resize(dim_out);
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
bool test_conv_int8(int n,
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
  std::unique_ptr<paddle::lite::KernelContext> ctx2(
      new paddle::lite::KernelContext);
#ifdef LITE_WITH_ARM
  auto& ctx_tmp1 = ctx1->As<paddle::lite::ARMContext>();
  ctx_tmp1.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cluster_id),
                      thread_num);
  auto& ctx_tmp2 = ctx1->As<paddle::lite::ARMContext>();
  ctx_tmp2.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cluster_id),
                      thread_num);
#else
  return true;
#endif
  ConvParam param_int8_out;
  ConvParam param_fp32_out;

  get_conv_param<PRECISION(kInt8)>(n,
                                   c,
                                   h,
                                   w,
                                   num_out,
                                   group,
                                   kernels,
                                   strides,
                                   pads,
                                   dilas,
                                   flag_bias,
                                   flag_relu,
                                   &param_int8_out);

  get_conv_param<PRECISION(kFloat)>(n,
                                    c,
                                    h,
                                    w,
                                    num_out,
                                    group,
                                    kernels,
                                    strides,
                                    pads,
                                    dilas,
                                    flag_bias,
                                    flag_relu,
                                    &param_fp32_out);

  auto dim_in = param_int8_out.x->dims();
  auto dim_out = param_int8_out.output->dims();
  auto dim_w = param_int8_out.filter->dims();

  if (dim_out[2] <= 0 || dim_out[3] <= 0) {
    release_param(&param_int8_out);
    release_param(&param_fp32_out);
    return true;
  }

  Tensor tin_fp32;
  Tensor weight_fp32;
  Tensor bias_fp32;
  Tensor tout_basic_fp32;
  Tensor tout_basic_int8;

  tin_fp32.Resize(dim_in);
  weight_fp32.Resize(dim_w);
  tout_basic_fp32.Resize(dim_out);
  tout_basic_int8.Resize(dim_out);

  paddle::lite::fill_tensor_rand(*param_int8_out.x, -127, 127);
  paddle::lite::fill_tensor_rand(*param_int8_out.filter, -127, 127);

  param_fp32_out.x->CopyDataFrom(*param_int8_out.x);
  param_fp32_out.filter->CopyDataFrom(*param_int8_out.filter);
  if (flag_bias) {
    auto dim_b = param_int8_out.bias->dims();
    bias_fp32.Resize(dim_b);
    paddle::lite::fill_tensor_rand(*param_int8_out.bias, -1.f, 1.f);
    param_fp32_out.bias->CopyDataFrom(*param_int8_out.bias);
    bias_fp32.CopyDataFrom(*param_int8_out.bias);
  }

  std::vector<float> scale_in{1.f / 127};
  std::vector<float> scale_out{c * kernels[0] * kernels[1] / (group * 127.f)};
  std::vector<float> scale_w(num_out, 1.f / 127);

  param_int8_out.input_scale = scale_in[0];
  param_int8_out.output_scale = scale_out[0];
  param_int8_out.weight_scale = scale_w;

  param_fp32_out.input_scale = scale_in[0];
  param_fp32_out.output_scale = scale_out[0];
  param_fp32_out.weight_scale = scale_w;

  auto din_int8 = param_int8_out.x->data<int8_t>();
  auto dw_int8 = param_int8_out.filter->data<int8_t>();
  auto din_fp32 = tin_fp32.mutable_data<float>();
  auto dw_fp32 = weight_fp32.mutable_data<float>();
  const float* dbias_fp32 = flag_bias ? bias_fp32.data<float>() : nullptr;

  paddle::lite::arm::math::int8_to_fp32(
      din_int8, din_fp32, scale_in.data(), 1, 1, dim_in.production());
  paddle::lite::arm::math::int8_to_fp32(dw_int8,
                                        dw_fp32,
                                        scale_w.data(),
                                        num_out,
                                        1,
                                        dim_w.production() / num_out);

  if (g_compare_result) {
    tout_basic_fp32.set_precision(PRECISION(kFloat));
    tout_basic_int8.set_precision(PRECISION(kInt8));
    paddle::lite::fill_tensor_const(tout_basic_fp32, 0.f);
    auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
    auto dout_baisc_int8 = tout_basic_int8.mutable_data<int8_t>();
    conv_basic<float, float>(din_fp32,
                             dout_basic_fp32,
                             dim_in[0],
                             dim_out[1],
                             dim_out[2],
                             dim_out[3],
                             dim_in[1],
                             dim_in[2],
                             dim_in[3],
                             dw_fp32,
                             dbias_fp32,
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
    paddle::lite::arm::math::fp32_to_int8(dout_basic_fp32,
                                          dout_baisc_int8,
                                          scale_out.data(),
                                          1,
                                          1,
                                          dim_out.production());
  }

  auto conv_int8_int8 =
      new paddle::lite::kernels::arm::ConvCompute<PRECISION(kInt8),
                                                  PRECISION(kInt8)>;
  conv_int8_int8->SetContext(std::move(ctx1));
  conv_int8_int8->SetParam(param_int8_out);
  auto conv_int8_fp32 =
      new paddle::lite::kernels::arm::ConvCompute<PRECISION(kInt8),
                                                  PRECISION(kFloat)>;
  conv_int8_fp32->SetContext(std::move(ctx2));
  conv_int8_fp32->SetParam(param_fp32_out);

  /// prepare for run
  conv_int8_int8->PrepareForRun();
  conv_int8_fp32->PrepareForRun();
  /// warm up
  for (int i = 0; i < g_warmup_iter; ++i) {
    conv_int8_int8->Launch();
  }

  double gops =
      2.0 * dim_out.production() * c * kernels[2] * kernels[3] / group;
  /// compute fp32 output
  lite::test::Timer t0;
  for (int i = 0; i < g_test_iter; ++i) {
    t0.start();
    conv_int8_fp32->Launch();
    t0.end();
  }
  LOG(INFO) << "int8 conv, fp32 output: output shape" << dim_out
            << ",running time, avg: " << t0.get_average_ms()
            << ", min time: " << t0.get_min_time()
            << ", total GOPS: " << 1e-9 * gops
            << " GOPS, avg GOPs: " << 1e-6 * gops / t0.get_average_ms()
            << " GOPs, max GOPs: " << 1e-6 * gops / t0.get_min_time();

  /// compute int8 output
  t0.clear();
  for (int i = 0; i < g_test_iter; ++i) {
    t0.start();
    conv_int8_int8->Launch();
    t0.end();
  }
  LOG(INFO) << "int8 conv, int8 output: output shape" << dim_out
            << ",running time, avg: " << t0.get_average_ms()
            << ", min time: " << t0.get_min_time()
            << ", total GOPS: " << 1e-9 * gops
            << " GOPS, avg GOPs: " << 1e-6 * gops / t0.get_average_ms()
            << " GOPs, max GOPs: " << 1e-6 * gops / t0.get_min_time();

  /// compare result fp32 output
  if (g_compare_result) {
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
        LOG(WARNING) << "saber result";
        print_tensor(*param_fp32_out.output);
        Tensor tdiff;
        tdiff.Resize(tout_basic_fp32.dims());
        tdiff.set_precision(PRECISION(kFloat));
        tensor_diff(tout_basic_fp32, *param_fp32_out.output, tdiff);
        print_tensor(tdiff);
        release_param(&param_int8_out);
        release_param(&param_fp32_out);
        return false;
      }
    }
    LOG(INFO) << "conv int8, output fp32 passed";
  }
  /// compare result int8 output
  if (g_compare_result) {
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
                     << ", after scale: " << ptr_basic_fp32[i] / scale_out[0];
          break;
        }
        if (ptr[i] != 0) {
          LOG(ERROR) << "basic float data: " << ptr_basic_fp32[i]
                     << ", after scale: " << ptr_basic_fp32[i] / scale_out[0];
          count += 1;
        }
      }
      check =
          check && count < std::max(10, static_cast<int>(0.01 * tdiff.numel()));
      if (!check) {
        LOG(WARNING) << "int8 basic result";
        print_tensor(tout_basic_int8);
        LOG(WARNING) << "int8 saber result";
        print_tensor(*param_int8_out.output);
        LOG(WARNING) << "int8 diff tensor";
        print_tensor(tdiff);
        release_param(&param_int8_out);
        release_param(&param_fp32_out);
        return false;
      }
    }
    LOG(INFO) << "conv_int8, output int8 passed";
  }

  release_param(&param_int8_out);
  release_param(&param_fp32_out);
  delete conv_int8_int8;
  delete conv_int8_fp32;

  return true;
}
#else
bool test_conv_int8(int n,
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
  return true;
}
#endif  // LITE_WITH_ARM

#if 1  /// 3x3dw
TEST(TestLite, test_conv_depthwise) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
      for (auto& c : {1, 3, 5, 8, 16, 32}) {
        for (auto& h : {1, 3, 8, 15, 19, 32, 38, 56, 75}) {
          for (auto& stride : {1, 2}) {
            for (auto& flag_bias : {false, true}) {
              for (auto& flag_relu : {false, true}) {
                //！ fix me, pad == 0 no pass
                for (auto& pad : {0, 1}) {
                  for (auto& th : {1, 2, 4}) {
                    int w = h;
                    if (!test_conv_int8(batch,
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
                          << "test int8 3x3 depthwise conv: batchsize: "
                          << batch << ", channel: " << c << ", h & w: " << h
                          << ", stride: " << stride << ", pad: " << pad
                          << ", bias: " << (flag_bias ? "true" : "false")
                          << ", relu: " << (flag_relu ? "true" : "false")
                          << ", threads: " << th << ", cluster: " << g_cluster
                          << " failed!!\n";
                      return;
                    }
                    LOG(INFO)
                        << "test int8 3x3 depthwise conv: batchsize: " << batch
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
#endif  /// 3x3dw

#if 0   /// 5x5dw
TEST(TestLite, test_conv_depthwise) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto& c : {1, 3, 5, 8, 16, 32}) {
    for (auto& h : {1, 3, 8, 15, 19, 32, 38, 56, 75}) {
    for (auto& stride : {1, 2}) {
    for (auto& flag_bias : {false, true}) {
    for (auto& flag_relu : {false, true}) {
    for (auto& pad : {0, 1, 2}) {
    for (auto& th : {1, 2, 4}) {
      int w = h;
      if (h == 1 && stride == 2) {
        continue;
      }
      if (!test_conv_int8(batch,
                          c,
                          h,
                          w,
                          c,
                          c,
                          {5, 5},
                          {stride, stride},
                          {pad, pad},
                          {1, 1},
                          flag_bias,
                          flag_relu,
                          th,
                          g_cluster)) {
        LOG(FATAL) << "test int8 5x5 depthwise conv: batchsize: "
                   << batch << ", channel: " << c << ", h & w: " << h
                   << ", stride: " << stride << ", pad: " << pad << ", bias: "
                   << (flag_bias? "true" : "false") << ", relu: "
                   << (flag_relu? "true" : "false") << ", threads: "
                   << th << ", cluster: " << g_cluster << " failed!!\n";
        return;
      }
      LOG(INFO) << "test int8 5x5 depthwise conv: batchsize: "
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
#endif  /// 5x5dw

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
                    if (!test_conv_int8(batch,
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
                      LOG(ERROR) << "test int8 1x1 conv: batchsize: " << batch
                                 << ", channel: " << c << ", h & w: " << h
                                 << ", bias: " << (flag_bias ? "true" : "false")
                                 << ", relu: " << (flag_relu ? "true" : "false")
                                 << ", threads: " << th
                                 << ", cluster: " << g_cluster << " failed!!\n";
                      return;
                    }
                    LOG(INFO) << "test int8 1x1 conv: batchsize: " << batch
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
#endif  /// conv1x1s1

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
      if (!test_conv_int8(batch,
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
        LOG(FATAL) << "test int8 3x3s1 conv: batchsize: " << batch
                   << ", channel: " << cin << ", h & w: " << h
                   << ", num_out: " << cout << ", pad: " << pad
                   << ", bias: " << (flag_bias ? "true" : "false")
                   << ", relu: " << (flag_relu ? "true" : "false")
                   << ", threads: " << th
                   << ", cluster: " << g_cluster << " failed!!\n";
        return;
      }
      LOG(INFO) << "test int8 3x3s1 conv: batchsize: " << batch
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
#endif  /// conv3x3s1

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
      if (!test_conv_int8(batch,
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
        LOG(FATAL) << "test int8 3x3s2 conv: batchsize: "
                   << batch << ", channel: " << cin
                   << ", h & w: " << h << ", num_out: " << cout
                   << ", bias: " << (flag_bias ? "true" : "false")
                   << ", relu: " << (flag_relu ? "true" : "false")
                   << ", threads: " <<  th
                   << ", cluster: " << g_cluster << " failed!!\n";
        return;
      }
      LOG(INFO) << "test int8 3x3s2 conv: batchsize: "
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
#endif  /// conv3x3s2

#if 0   /// random param conv
TEST(TestLite, test_conv_rand) {
  if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto &cin : {1, 3, 8, 16, 24, 32}) {
    for (auto& cout : {1, 5, 8, 16, 24, 32}) {
    for (auto &g_div : {1, 2}) {
    for (auto &h : {1, 3, 19, 28, 32, 41, 56, 75}) {
    for (auto& kw : {1, 2, 3}) {
    for (auto& kh : {1, 2, 3}) {
    for (auto& stride : {1, 2}) {
    for (auto& dila : {1, 2}) {
    for (auto &pad : {0, 1, 2}) {
    for (auto &flag_bias : {false, true}) {
    for (auto &flag_relu : {false, true}) {
    for (auto &th : {1, 2, 4}) {
      int w = h;
      int g = g_div;
      if (cin % g != 0 || cout % g != 0) {
        continue;
      }
      auto flag = test_conv_int8(batch,
                                 cin,
                                 h,
                                 w,
                                 cout,
                                 g,
                                 {kh, kw},
                                 {stride, stride},
                                 {pad, pad},
                                 {dila, dila},
                                 flag_bias,
                                 flag_relu,
                                 th,
                                 g_cluster);
      if (flag) {
        LOG(INFO) << "test fp32 conv: batchsize: " << batch
                  << ", channel: " << cin << ", h: " << h
                  << ", w: " << w << ", num_out: " << cout
                  << ", group: " << g << ", kw: " << kw << ", kh: " << kh
                  << ", pad: " << pad << ", stride: " << stride
                  << ", dila_: " << dila
                  << ", bias: " << (flag_bias ? "true" : "false")
                  << ", relu: " << (flag_relu ? "true" : "false")
                  << ", threads: " << th << ", cluster: " << g_cluster
                  << " passed!!\n";
      } else {
        LOG(FATAL) << "test fp32 conv: batchsize: " << batch
                   << ", channel: " << cin << ", h: " << h
                   << ", w: " << w << ", num_out: " << cout
                   << ", group: " << g << ", kw: " << kw << ", kh: " << kh
                   << ", pad: " << pad << ", stride: " << stride
                   << ", dila_: " << dila
                   << ", bias: " << (flag_bias ? "true" : "false")
                   << ", relu: " << (flag_relu ? "true" : "false")
                   << ", threads: " << th << ", cluster: " << g_cluster
                   << " failed!!\n";
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

#if 0   /// custom
TEST(TestLite, test_conv_int8_custom_size) {
  auto flag = test_conv_int8(g_num,
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
    LOG(INFO) << "test int8 conv: batchsize: " << g_num
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
    LOG(FATAL) << "test int8 conv: batchsize: " << g_num
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
