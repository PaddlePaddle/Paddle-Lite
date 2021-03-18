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
#include "lite/tests/utils/print_info.h"
#include "lite/tests/utils/tensor_utils.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/fc_compute.h"
#endif  // LITE_WITH_ARM

DEFINE_int32(in_num_col_dims, 1, "input width");
DEFINE_int32(M, 512, "gemm: M");
DEFINE_int32(N, 512, "gemm: N");
DEFINE_int32(K, 512, "gemm: K");

typedef paddle::lite::operators::FcParam FcParam;

DDim compute_out_dim(const DDim& dim_in,
                     const DDim& wdim,
                     int in_num_col_dims) {
  std::vector<int64_t> out_dim;
  out_dim.resize(in_num_col_dims + 1);
  for (int i = 0; i < in_num_col_dims; ++i) {
    out_dim[i] = dim_in[i];
  }
  out_dim[in_num_col_dims] = wdim[1];
  return DDim(out_dim);
}

void AddBias(float* out, const float* bias, int num, int channel) {
  int remain = channel;
  for (int j = 0; j < num; ++j) {
    const float* ptr_bias = bias;
    float* ptr_out = out + j * channel;
    for (int i = 0; i < remain; ++i) {
      *(ptr_out++) += *(ptr_bias++);
    }
  }
}

#ifdef LITE_WITH_ARM
void test_fc_fp16(const DDim in_dim,
                  const DDim weight_dim,
                  const DDim bias_dim,
                  int in_num_col_dims,
                  bool flag_bias,
                  const std::vector<int>& thread_num,
                  const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  FcParam param;
  param.input = new Tensor;
  param.input->set_precision(PRECISION(kFP16));
  param.w = new Tensor;
  param.w->Resize(weight_dim);
  param.w->set_precision(PRECISION(kFP16));
  param.in_num_col_dims = in_num_col_dims;
  if (flag_bias) {
    param.bias = new Tensor;
    param.bias->Resize(bias_dim);
    param.bias->set_precision(PRECISION(kFP16));
  }

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFP16));
  Tensor filter_fp32;
  filter_fp32.Resize(weight_dim);
  filter_fp32.set_precision(PRECISION(kFloat));
  auto a_ptr = filter_fp32.mutable_data<float>();
  auto b_ptr = param.w->mutable_data<float16_t>();
  fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.w->numel());
  // fill_data_const<float16_t>(b_ptr, -1.f, param.w->numel());
  fp16_to_float(param.w->data<float16_t>(), a_ptr, param.w->numel());

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
  auto wptr = param.w->data<float16_t>();
  auto bias_ptr = flag_bias ? param.bias->data<float16_t>() : nullptr;
  int M = in_dim.count(0, in_num_col_dims);
  CHECK_EQ(weight_dim[0], in_dim.count(in_num_col_dims, in_dim.size()));
  int K = weight_dim[0];
  int N = weight_dim[1];

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::FcCompute<PRECISION(kFP16), PRECISION(kFP16)>
          fc;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      DDim dim_out = compute_out_dim(in_dim, weight_dim, in_num_col_dims);
      if (dim_out[2] < 1 || dim_out[3] < 1) {
        return;
      }
      param.output->Resize(dim_out);
      /// set param and context
      fc.SetParam(param);
      fc.SetContext(std::move(ctx1));
      /// prepare for run
      fc.PrepareForRun();

      param.input->Resize(in_dim);
      param.output->Resize(in_dim);

      Tensor x_fp32;
      x_fp32.Resize(in_dim);
      x_fp32.set_precision(PRECISION(kFloat));
      auto a_ptr = x_fp32.mutable_data<float>();
      auto b_ptr = param.input->mutable_data<float16_t>();
      // fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.x->numel());
      fill_data_const<float16_t>(b_ptr, -1.f, param.input->numel());
      fp16_to_float(
          param.input->data<float16_t>(), a_ptr, param.input->numel());
      auto din = param.input->data<float16_t>();
      auto din_fp32 = x_fp32.data<float>();

      Tensor tout_basic;
      if (FLAGS_check_result) {
        Tensor tout_basic_fp32;
        tout_basic_fp32.set_precision(PRECISION(kFloat));
        tout_basic.set_precision(PRECISION(kFP16));
        tout_basic_fp32.Resize(in_dim);
        tout_basic.Resize(in_dim);

        auto dout_basic = tout_basic.mutable_data<float16_t>();
        auto dout_basic_fp16 = tout_basic.mutable_data<float16_t>();
        auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
        auto bias_fp32_ptr = flag_bias ? bias_fp32.data<float>() : nullptr;
        auto filter_fp32_ptr = filter_fp32.data<float>();

        fill_data_const<float>(dout_basic_fp32, 0.f, tout_basic_fp32.numel());
        fill_data_const<float16_t>(dout_basic, 0.f, tout_basic.numel());
        if (M == 1) {
          basic_gemv<float, float>(N,
                                   K,
                                   din_fp32,
                                   filter_fp32_ptr,
                                   bias_fp32_ptr,
                                   dout_basic_fp32,
                                   1.f,
                                   0.f,
                                   true,
                                   flag_bias,
                                   false);
        } else {
          basic_gemm<float, float>(false,
                                   false,
                                   M,
                                   N,
                                   K,
                                   1.f,
                                   din_fp32,
                                   K,
                                   filter_fp32_ptr,
                                   N,
                                   0.f,
                                   dout_basic_fp32,
                                   N,
                                   bias_fp32_ptr,
                                   false,
                                   false);
          if (flag_bias) {
            AddBias(dout_basic_fp32, bias_fp32_ptr, M, N);
          }
        }
        // fp32->fp16
        float_to_fp16(dout_basic_fp32, dout_basic, tout_basic.numel());
      }
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        fc.Launch();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        fc.Launch();
        t0.Stop();
      }

      VLOG(4) << "fc fp16: input shape: " << in_dim
              << ", in_num_col_dims: " << in_num_col_dims << ", M: " << M
              << ", N: " << N << ", K: " << K
              << ", running time, avg: " << t0.LapTimes().Avg()
              << ", min time: " << t0.LapTimes().Min();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        auto basic_ptr = tout_basic.data<float16_t>();
        auto saber_ptr = param.output->data<float16_t>();
        Tensor tdiff;
        tdiff.Resize(tout_basic.dims());
        tdiff.set_precision(PRECISION(kFP16));
        auto ptr = tdiff.mutable_data<float16_t>();
        data_diff(
            basic_ptr, saber_ptr, ptr, tout_basic.numel(), max_ratio, max_diff);
        print_diff_info(max_diff, max_ratio);
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 4e-3f) {
            int64_t size = tout_basic.numel();
            int64_t width = in_dim[3];
            print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
            LOG(FATAL) << "test fp16 fc: input: " << in_dim << ", M: " << M
                       << ", N: " << N << ", K: " << K << ", threads: " << th
                       << ", power_mode: " << cls << " failed!!\n";
          }
        }
      }
      LOG(INFO) << "test fp16 fc: input: " << in_dim << ", M: " << M
                << ", N: " << N << ", K: " << K << ", threads: " << th
                << ", power_mode: " << cls << " successed!!\n";
    }
  }

  delete param.input;
  delete param.output;
}

#else
void test_fc_fp16(const DDim in_dim,
                  int in_num_col_dims,
                  const std::vector<int>& thread_num,
                  const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param fc
TEST(TestFcRand, test_fc_rand) {
  if (FLAGS_basic_test) {
    for (auto& m : {1, 3, 16}) {
      for (auto& n : {1, 4, 16, 128, 256, 1024}) {
        for (auto& k : {1, 16, 128, 1024}) {
          for (auto flag_bias : {false, true}) {
            DDim in_dim{{m, k}};
            DDim wei_dim{{k, n}};
            DDim bias_dim{{flag_bias ? n : 0}};
            test_fc_fp16(in_dim,
                         wei_dim,
                         bias_dim,
                         1,
                         flag_bias,
                         {4},
                         {FLAGS_power_mode});
          }
        }
      }
    }
  }
}
#endif  /// random param conv

#if 1  /// custom
TEST(TesSoftmaxCustom, test_softmax_fp16_custom_size) {
  test_fc_fp16(DDim({FLAGS_M, FLAGS_K}),
               DDim({FLAGS_K, FLAGS_N}),
               DDim({FLAGS_flag_bias ? FLAGS_N : 0}),
               FLAGS_in_num_col_dims,
               FLAGS_flag_bias,
               {FLAGS_threads},
               {FLAGS_power_mode});
}
#endif  // custom
