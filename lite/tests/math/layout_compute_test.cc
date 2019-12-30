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
#include "lite/kernels/arm/layout_compute.h"
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

DEFINE_bool(flag_nchw, true, "do nchw to nhwc");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::LayoutParam LayoutParam;

using paddle::lite::profile::Timer;

#define IN(n, c, h, w)                                 \
  input_data[w + h * input_w + c * input_h * input_w + \
             n * input_c * input_h * input_w]
#define OUT(n, c, h, w)                                    \
  output_data[w + h * output_w + c * output_h * output_w + \
              n * output_c * output_h * output_w]

template <typename Dtype>
void nchw2nhwc_ref(const Tensor* input, Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_c = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, h, w, c) = IN(n, c, h, w);
        }
      }
    }
  }
}
#undef IN
#undef OUT

#define IN(n, h, w, c)                                 \
  input_data[c + w * input_c + h * input_w * input_c + \
             n * input_h * input_w * input_c]
#define OUT(n, h, w, c)                                    \
  output_data[c + w * output_c + h * output_w * output_c + \
              n * output_h * output_w * output_c]
template <typename Dtype>
void nhwc2nchw_ref(const Tensor* input, Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_h = input->dims()[1];
  int input_w = input->dims()[2];
  int input_c = input->dims()[3];
  int output_h = output->dims()[1];
  int output_w = output->dims()[2];
  int output_c = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, c, h, w) = IN(n, h, w, c);
        }
      }
    }
  }
}

#ifdef LITE_WITH_ARM
void test_layout_fp32_nchw(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  LayoutParam param;
  param.x = new Tensor;
  const_cast<Tensor*>(param.x)->set_precision(PRECISION(kFloat));

  param.y = new Tensor;
  param.y->set_precision(PRECISION(kFloat));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::NCHWToNHWCCompute<PRECISION(kFloat)> layout;
      DDim dim_out({dim_in[0], dim_in[2], dim_in[3], dim_in[1]});

      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      const_cast<Tensor*>(param.x)->Resize(dim_in);
      param.y->Resize(dim_out);

      layout.SetParam(param);

      paddle::lite::fill_tensor_rand(
          *(const_cast<Tensor*>(param.x)), -1.f, 1.f);
      //   paddle::lite::fill_tensor_const(*param.x, 1.f);

      auto din = param.x->data<float>();

      Tensor tout_basic;

      if (FLAGS_check_result) {
        tout_basic.set_precision(PRECISION(kFloat));
        tout_basic.Resize(dim_out);
        fill_tensor_const(tout_basic, 0.f);
        auto dout_basic = tout_basic.mutable_data<float>();
        nchw2nhwc_ref<float>(param.x, &tout_basic);
      }
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        layout.Run();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        layout.Run();
        t0.Stop();
      }
      double gops = 2.0 * dim_out.production();
      LOG(INFO) << "layout fp32: input shape: " << dim_in << ", output shape"
                << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
                << ", min time: " << t0.LapTimes().Min()
                << ", total GOPS: " << 1e-9 * gops
                << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, *param.y, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 5e-4f) {
            LOG(WARNING) << "din";
            print_tensor(*(const_cast<Tensor*>(param.x)));
            LOG(WARNING) << "basic result";
            print_tensor(tout_basic);
            LOG(WARNING) << "lite result";
            print_tensor(*param.y);
            Tensor tdiff;
            tdiff.Resize(tout_basic.dims());
            tdiff.set_precision(PRECISION(kFloat));
            tensor_diff(tout_basic, *param.y, tdiff);
            print_tensor(tdiff);
            LOG(FATAL) << "test fp32 layout: input: " << dim_in
                       << ", output: " << dim_out << ", flag_nchw: "
                       << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                       << ", threads: " << th << ", power_mode: " << cls
                       << " failed!!\n";
          }
        }
        LOG(INFO) << "test fp32 layout: input: " << dim_in
                  << ", output: " << dim_out
                  << ", flag_nchw: " << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.y;
}
void test_layout_fp32_nhwc(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif

  LayoutParam param;
  param.x = new Tensor;
  const_cast<Tensor*>(param.x)->set_precision(PRECISION(kFloat));

  param.y = new Tensor;
  param.y->set_precision(PRECISION(kFloat));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::NHWCToNCHWCompute<PRECISION(kFloat)> layout;
      // n h w c == n c h w
      DDim dim_out({dim_in[0], dim_in[3], dim_in[1], dim_in[2]});

      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      const_cast<Tensor*>(param.x)->Resize(dim_in);
      param.y->Resize(dim_out);

      layout.SetParam(param);

      paddle::lite::fill_tensor_rand(
          *(const_cast<Tensor*>(param.x)), -1.f, 1.f);
      //   paddle::lite::fill_tensor_const(*param.x, 1.f);

      auto din = param.x->data<float>();

      Tensor tout_basic;

      if (FLAGS_check_result) {
        tout_basic.set_precision(PRECISION(kFloat));
        tout_basic.Resize(dim_out);
        fill_tensor_const(tout_basic, 0.f);
        auto dout_basic = tout_basic.mutable_data<float>();
        nhwc2nchw_ref<float>(param.x, &tout_basic);
      }
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        layout.Run();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        layout.Run();
        t0.Stop();
      }
      double gops = 2.0 * dim_out.production();
      LOG(INFO) << "layout fp32: input shape: " << dim_in << ", output shape"
                << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
                << ", min time: " << t0.LapTimes().Min()
                << ", total GOPS: " << 1e-9 * gops
                << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, *param.y, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 5e-4f) {
            LOG(WARNING) << "din";
            print_tensor(*(const_cast<Tensor*>(param.x)));
            LOG(WARNING) << "basic result";
            print_tensor(tout_basic);
            LOG(WARNING) << "lite result";
            print_tensor(*param.y);
            Tensor tdiff;
            tdiff.Resize(tout_basic.dims());
            tdiff.set_precision(PRECISION(kFloat));
            tensor_diff(tout_basic, *param.y, tdiff);
            print_tensor(tdiff);
            LOG(FATAL) << "test fp32 layout: input: " << dim_in
                       << ", output: " << dim_out << ", flag_nchw: "
                       << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                       << ", threads: " << th << ", power_mode: " << cls
                       << " failed!!\n";
          }
        }
        LOG(INFO) << "test fp32 layout: input: " << dim_in
                  << ", output: " << dim_out
                  << ", flag_nchw: " << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.y;
}
void test_layout_int8_nchw(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif

  LayoutParam param;
  param.x = new Tensor;
  const_cast<Tensor*>(param.x)->set_precision(PRECISION(kInt8));

  param.y = new Tensor;
  param.y->set_precision(PRECISION(kInt8));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::NCHWToNHWCCompute<PRECISION(kInt8)> layout;
      DDim dim_out({dim_in[0], dim_in[2], dim_in[3], dim_in[1]});

      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      const_cast<Tensor*>(param.x)->Resize(dim_in);
      param.y->Resize(dim_out);

      layout.SetParam(param);

      paddle::lite::fill_tensor_rand(*(const_cast<Tensor*>(param.x)));
      //   paddle::lite::fill_tensor_const(*param.x, 1.f);

      auto din = param.x->data<int8_t>();

      Tensor tout_basic;

      if (FLAGS_check_result) {
        tout_basic.set_precision(PRECISION(kInt8));
        tout_basic.Resize(dim_out);
        fill_tensor_const(tout_basic, 0);
        auto dout_basic = tout_basic.mutable_data<int8_t>();
        nchw2nhwc_ref<int8_t>(param.x, &tout_basic);
      }
      LOG(INFO) << "saber compute";
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        layout.Run();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        layout.Run();
        t0.Stop();
      }
      LOG(INFO) << "saber compute end";
      double gops = 2.0 * dim_out.production();
      LOG(INFO) << "layout int8: input shape: " << dim_in << ", output shape"
                << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
                << ", min time: " << t0.LapTimes().Min()
                << ", total GOPS: " << 1e-9 * gops
                << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, *param.y, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 5e-4f) {
            LOG(WARNING) << "din";
            print_tensor(*(const_cast<Tensor*>(param.x)));
            LOG(WARNING) << "basic result";
            print_tensor(tout_basic);
            LOG(WARNING) << "lite result";
            print_tensor(*param.y);
            Tensor tdiff;
            tdiff.Resize(tout_basic.dims());
            tdiff.set_precision(PRECISION(kInt8));
            tensor_diff(tout_basic, *param.y, tdiff);
            print_tensor(tdiff);
            LOG(FATAL) << "test int8 layout: input: " << dim_in
                       << ", output: " << dim_out << ", flag_nchw: "
                       << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                       << ", threads: " << th << ", power_mode: " << cls
                       << " failed!!\n";
          }
        }
        LOG(INFO) << "test int8 layout: input: " << dim_in
                  << ", output: " << dim_out
                  << ", flag_nchw: " << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.y;
}
void test_layout_int8_nhwc(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif

  LayoutParam param;
  param.x = new Tensor;
  const_cast<Tensor*>(param.x)->set_precision(PRECISION(kInt8));

  param.y = new Tensor;
  param.y->set_precision(PRECISION(kInt8));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::NHWCToNCHWCompute<PRECISION(kInt8)> layout;
      // n h w c == n c h w
      DDim dim_out({dim_in[0], dim_in[3], dim_in[1], dim_in[2]});

      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      const_cast<Tensor*>(param.x)->Resize(dim_in);
      param.y->Resize(dim_out);

      layout.SetParam(param);

      paddle::lite::fill_tensor_rand(*(const_cast<Tensor*>(param.x)));
      //   paddle::lite::fill_tensor_const(*param.x, 1.f);

      auto din = param.x->data<int8_t>();

      Tensor tout_basic;

      if (FLAGS_check_result) {
        tout_basic.set_precision(PRECISION(kInt8));
        tout_basic.Resize(dim_out);
        fill_tensor_const(tout_basic, 0.f);
        auto dout_basic = tout_basic.mutable_data<int8_t>();
        nhwc2nchw_ref<int8_t>(param.x, &tout_basic);
      }
      LOG(INFO) << "saber compute";
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        layout.Run();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        layout.Run();
        t0.Stop();
      }
      LOG(INFO) << "run";
      double gops = 2.0 * dim_out.production();
      LOG(INFO) << "layout int8: input shape: " << dim_in << ", output shape"
                << dim_out << ",running time, avg: " << t0.LapTimes().Avg()
                << ", min time: " << t0.LapTimes().Min()
                << ", total GOPS: " << 1e-9 * gops
                << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, *param.y, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff
                  << ", max ratio: " << max_ratio;
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 5e-4f) {
            LOG(WARNING) << "din";
            print_tensor(*(const_cast<Tensor*>(param.x)));
            LOG(WARNING) << "basic result";
            print_tensor(tout_basic);
            LOG(WARNING) << "lite result";
            print_tensor(*param.y);
            Tensor tdiff;
            tdiff.Resize(tout_basic.dims());
            tdiff.set_precision(PRECISION(kInt8));
            tensor_diff(tout_basic, *param.y, tdiff);
            print_tensor(tdiff);
            LOG(FATAL) << "test int8 layout: input: " << dim_in
                       << ", output: " << dim_out << ", flag_nchw: "
                       << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                       << ", threads: " << th << ", power_mode: " << cls
                       << " failed!!\n";
          }
        }
        LOG(INFO) << "test int8 layout: input: " << dim_in
                  << ", output: " << dim_out
                  << ", flag_nchw: " << (flag_nchw ? "nchw2nhwc" : "nhwc2nchw")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.y;
}
#else
void test_layout_fp32_nchw(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {}
void test_layout_fp32_nhwc(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {}
void test_layout_int8_nchw(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {}
void test_layout_int8_nhwc(DDim dim_in,
                           bool flag_nchw,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  //
TEST(TestLayout, test_Layout_fp32) {
  if (FLAGS_basic_test) {
    for (auto n : {1, 3}) {
      for (auto c : {1, 3, 5, 32}) {
        for (auto h : {3, 16, 20, 32}) {
          for (auto w : {3, 4, 32, 112}) {
            for (auto nchw2nhwc : {true, false}) {
              DDim dim_in({n, c, h, w});
              if (nchw2nhwc) {
                LOG(INFO) << "NCHW2NHWC";
                test_layout_fp32_nchw(
                    dim_in, nchw2nhwc, {1, 2, 4}, {FLAGS_power_mode});
              } else {
                LOG(INFO) << "NHWC2NCHW";
                test_layout_fp32_nhwc(
                    dim_in, nchw2nhwc, {1, 2, 4}, {FLAGS_power_mode});
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if 1
TEST(TestLayout, test_Layout_int8) {
  if (FLAGS_basic_test) {
    for (auto n : {1, 3}) {
      for (auto c : {1, 3, 5, 32}) {
        for (auto h : {3, 16, 20, 32}) {
          for (auto w : {3, 4, 32, 112}) {
            for (auto nchw2nhwc : {true, false}) {
              DDim dim_in({n, c, h, w});
              if (nchw2nhwc) {
                LOG(INFO) << "NCHW2NHWC int8";
                test_layout_int8_nchw(
                    dim_in, nchw2nhwc, {1, 2, 4}, {FLAGS_power_mode});
              } else {
                LOG(INFO) << "NHWC2NCHW int8";
                test_layout_int8_nhwc(
                    dim_in, nchw2nhwc, {1, 2, 4}, {FLAGS_power_mode});
              }
            }
          }
        }
      }
    }
  }
}
#endif

#if 1  /// custom
TEST(TestLayoutCustom, test_Layout_custom_size) {
  test_layout_fp32_nchw(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      true,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif  // custom
