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
#include "lite/kernels/arm/pool_compute.h"
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

DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h, 1, "pad height");
DEFINE_int32(pad_w, 1, "pad width");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");

DEFINE_bool(ceil_mode, true, "do ceil_mode");
DEFINE_bool(flag_global, true, "global pooling");
DEFINE_bool(exclusive, true, "do exclusive");
DEFINE_bool(adaptive, false, "no do adaptive");
DEFINE_bool(use_quantizer, false, "no do use_quantizer");

DEFINE_string(pooling_type, "max", "do max pooling");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::PoolParam PoolParam;
using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::PoolParam& param) {
  DDim dim_out = dim_in;
  auto kernel_h = param.ksize[0];
  auto kernel_w = param.ksize[1];
  auto h = dim_in[2];
  auto w = dim_in[3];
  auto paddings = *param.paddings;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  bool ceil_mode = param.ceil_mode;
  bool flag_global = param.global_pooling;
  int hout = 1;
  int wout = 1;
  if (!flag_global) {
    if (!ceil_mode) {
      hout = (h - kernel_h + paddings[0] + paddings[1]) / stride_h + 1;
      wout = (w - kernel_w + paddings[2] + paddings[3]) / stride_w + 1;
    } else {
      hout =
          (h - kernel_h + paddings[0] + paddings[1] + stride_h - 1) / stride_h +
          1;
      wout =
          (w - kernel_w + paddings[2] + paddings[3] + stride_w - 1) / stride_w +
          1;
    }
  }
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

void pooling_basic(const float* din,
                   float* dout,
                   int num,
                   int chout,
                   int hout,
                   int wout,
                   int chin,
                   int hin,
                   int win,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool global_pooling,
                   bool exclusive,
                   bool adaptive,
                   bool ceil_mode,
                   bool use_quantizer,
                   const std::string& pooling_type) {
  // no need to pad input tensor, border is zero pad inside this function
  memset(dout, 0, num * chout * hout * wout * sizeof(float));
  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;
  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            float tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      // Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += din_ch[i];
          }
          dout_batch[c] = sum / size_channel_in;
        }
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  } else {
    for (int ind_n = 0; ind_n < num; ++ind_n) {
#pragma omp parallel for
      for (int ind_c = 0; ind_c < chin; ++ind_c) {
        for (int ind_h = 0; ind_h < hout; ++ind_h) {
          int sh = ind_h * stride_h;
          int eh = sh + kernel_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > hin ? hin : eh - pad_h;
          for (int ind_w = 0; ind_w < wout; ++ind_w) {
            int sw = ind_w * stride_w;
            int ew = sw + kernel_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > win ? win : ew - pad_w;
            float result = static_cast<float>(0);
            int dst_ind = (ind_n * chout + ind_c) * size_channel_out +
                          ind_h * wout + ind_w;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_ind =
                    (ind_n * chin + ind_c) * size_channel_in + kh * win + kw;
                if (kh == sh && kw == sw) {
                  result = din[src_ind];
                } else {
                  if (pooling_type == "max") {
                    result = result >= din[src_ind] ? result : din[src_ind];
                  } else if (pooling_type == "avg") {
                    result += din[src_ind];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                int div = (ew - sw) * (eh - sh);
                div = div > 0 ? div : 1;
                result /= div;
              } else {
                int bh = kernel_h;
                int bw = kernel_w;
                if (ew == win) {
                  bw = (sw + kernel_w) >= (win + paddings[3])
                           ? (win + paddings[3])
                           : (sw + kernel_w);
                  bw -= sw;
                  if ((sw - pad_w) < 0 &&
                      (sw + kernel_w) > (win + paddings[3])) {
                    bw += pad_w;
                  }
                }
                if (eh == hin) {
                  bh = (sh + kernel_h) >= (hin + paddings[1])
                           ? (hin + paddings[1])
                           : (sh + kernel_h);
                  bh -= sh;
                  if ((sh - pad_h) < 0 &&
                      (sh + kernel_h) > (hin + paddings[1])) {
                    bh += pad_h;
                  }
                }
                result /= bh * bw;
              }
            }
            dout[dst_ind] = result;
          }
        }
      }
    }
  }
}

#ifdef LITE_WITH_ARM
void test_pool_fp32(const std::vector<DDim>& input_dims,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    bool ceil_mode,
                    bool flag_global,
                    bool exclusive,
                    bool adaptive,
                    bool use_quantizer,
                    std::string pooling_type,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  PoolParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFloat));
  param.ksize = ksize;

  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.ceil_mode = ceil_mode;
  param.global_pooling = flag_global;
  param.pooling_type = pooling_type;
  param.exclusive = exclusive;
  param.adaptive = adaptive;
  param.use_quantizer = use_quantizer;

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFloat));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::PoolCompute pool;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      pool.SetParam(param);
      pool.SetContext(std::move(ctx1));
      /// prepare for run
      pool.PrepareForRun();

      for (auto& dim_in : input_dims) {
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
          LOG(INFO) << "basic compute";
          tout_basic.set_precision(PRECISION(kFloat));
          tout_basic.Resize(dim_out);
          fill_tensor_const(tout_basic, 0.f);
          auto dout_basic = tout_basic.mutable_data<float>();
          pooling_basic(din,
                        dout_basic,
                        dim_in[0],
                        dim_out[1],
                        dim_out[2],
                        dim_out[3],
                        dim_in[1],
                        dim_in[2],
                        dim_in[3],
                        ksize,
                        strides,
                        pads,
                        flag_global,
                        exclusive,
                        adaptive,
                        ceil_mode,
                        use_quantizer,
                        pooling_type);
        }
        LOG(INFO) << "lite compute";
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          pool.Launch();
        }
        /// compute
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          pool.Launch();
          t0.Stop();
        }

        double gops = 2.0 * dim_out.production() * ksize[0] * ksize[1];
        LOG(INFO) << "pool fp32: input shape: " << dim_in << ", output shape"
                  << dim_out << ", running time, avg: " << t0.LapTimes().Avg()
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
              LOG(WARNING) << "din";
              print_tensor(*param.x);
              LOG(WARNING) << "basic result";
              print_tensor(tout_basic);
              LOG(WARNING) << "lite result";
              print_tensor(*param.output);
              Tensor tdiff;
              tdiff.Resize(tout_basic.dims());
              tdiff.set_precision(PRECISION(kFloat));
              tensor_diff(tout_basic, *param.output, tdiff);
              print_tensor(tdiff);
              LOG(FATAL) << "test fp32 pool: input: " << dim_in
                         << ", output: " << dim_out
                         << ", kernel dim: " << ksize[0] << ", " << ksize[1]
                         << ", pad: " << pads[0] << ", " << pads[1] << ", "
                         << pads[2] << ", " << pads[3]
                         << ", stride: " << strides[0] << ", " << strides[1]
                         << ", global_pooling: "
                         << (flag_global ? "global" : "false")
                         << ", pooling_type: " << pooling_type
                         << ", ceil_mode: " << (ceil_mode ? "true" : "false")
                         << ", exclusive: " << (exclusive ? "true" : "false")
                         << ", threads: " << th << ", power_mode: " << cls
                         << " failed!!\n";
            }
          }
        }
        LOG(INFO) << "test fp32 pool: input: " << dim_in
                  << ", output: " << dim_out << ", kernel dim: " << ksize[0]
                  << ", " << ksize[1] << ", pad: " << pads[0] << ", " << pads[1]
                  << ", " << pads[2] << ", " << pads[3]
                  << ", stride: " << strides[0] << ", " << strides[1]
                  << ", global_pooling: " << (flag_global ? "global" : "false")
                  << ", pooling_type: " << pooling_type
                  << ", ceil_mode: " << (ceil_mode ? "true" : "false")
                  << ", exclusive: " << (exclusive ? "true" : "false")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.output;
}
#else
void test_pool_fp32(const std::vector<DDim>& input_dims,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    bool ceil_mode,
                    bool flag_global,
                    bool exclusive,
                    bool adaptive,
                    bool use_quantizer,
                    std::string pooling_type,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param pool
TEST(TestPoolRand, test_pool_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 16}) {
      for (auto& kw : {1, 2, 3}) {
        for (auto& kh : {1, 2, 3}) {
          for (auto& stride : {1, 2}) {
            for (auto& pad_top : {0, 1, 2}) {
              for (auto& pad_bottom : {0, 1, 2}) {
                for (auto& pad_left : {0, 1, 2}) {
                  for (auto& pad_right : {0, 1, 2}) {
                    for (auto& flag_global : {false, true}) {
                      for (auto& exclusive : {false, true}) {
                        for (auto& ceil_mode : {false, true}) {
                          for (auto& pooling_type : {"max", "avg"}) {
                            bool adaptive = false;
                            bool use_quantizer = false;
                            std::vector<DDim> dims;
                            for (auto& batch : {1, 2}) {
                              for (auto& h : {1, 2, 3, 4, 11, 19, 32, 28}) {
                                dims.push_back(DDim({batch, cin, h, h}));
                              }
                            }
                            test_pool_fp32(
                                dims,
                                {kh, kw},
                                {stride, stride},
                                {pad_top, pad_bottom, pad_left, pad_right},
                                ceil_mode,
                                flag_global,
                                exclusive,
                                adaptive,
                                use_quantizer,
                                pooling_type,
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
#endif  /// random param conv

#if 1  /// custom
TEST(TesPoolCustom, test_pool_fp32_custom_size) {
  test_pool_fp32(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      {FLAGS_kernel_h, FLAGS_kernel_w},
      {FLAGS_stride_h, FLAGS_stride_w},
      {FLAGS_pad_h, FLAGS_pad_h, FLAGS_pad_w, FLAGS_pad_w},
      FLAGS_ceil_mode,
      FLAGS_flag_global,
      FLAGS_exclusive,
      FLAGS_adaptive,
      FLAGS_use_quantizer,
      FLAGS_pooling_type,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif  // custom
