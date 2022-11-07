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
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#ifdef LITE_WITH_ARM
#include "lite/backends/arm/math/funcs.h"
#endif  // LITE_WITH_ARM
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/print_info.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;

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

DEFINE_int32(M, 512, "gemv: M");
DEFINE_int32(N, 512, "gemv: N");

DEFINE_bool(traA, false, "gemv: A transpose");

DEFINE_int32(flag_act, 0, "do act");
DEFINE_bool(flag_bias, false, "with bias");
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_double(clipped_coef, 6.0, "clipped relu coef");

bool test_sgemv_fp16(bool tra,
                     int m,
                     int n,
                     bool has_bias,
                     int flag_act,
                     int cls,
                     int ths,
                     float six = 6.f,
                     float alpha = 1.f) {
  Tensor ta;
  Tensor tb;
  Tensor tc;
  Tensor tc_basic;
  Tensor tc_backup;
  Tensor tbias;
  int size_a = m * n;
  int size_b = n;
  int size_c = m;

  ta.Resize({size_a});
  tb.Resize({size_b});
  tc.Resize({size_c});
  tc_basic.Resize({size_c});
  tc_backup.Resize({size_c});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kFP16));
  tb.set_precision(PRECISION(kFP16));
  tc.set_precision(PRECISION(kFP16));
  tc_basic.set_precision(PRECISION(kFP16));
  tc_backup.set_precision(PRECISION(kFP16));
  tbias.set_precision(PRECISION(kFP16));

  auto da = ta.mutable_data<float16_t>();
  auto db = tb.mutable_data<float16_t>();
  auto dc = tc.mutable_data<float16_t>();
  auto dc_basic = tc_basic.mutable_data<float16_t>();
  auto dc_backup = tc_backup.mutable_data<float16_t>();
  auto dbias = tbias.mutable_data<float16_t>();

  fill_data_rand<float16_t>(da, -1.f, 1.f, size_a);
  // fill_data_const<float16_t>(da, 1.f, size_a);

  fill_data_rand<float16_t>(db, -1.f, 1.f, size_b);
  // fill_data_const<float16_t>(db, 1.f, size_b);

  fill_data_rand<float16_t>(dbias, -1.f, 1.f, m);
  // fill_data_const<float16_t>(dbias, -1.f, m);
  // fill_data_rand<float16_t>(dc, -1.f, 1.f, size_c);
  fill_data_const<float16_t>(dc, 1.f, size_c);

  memcpy(dc_basic, dc, sizeof(float16_t) * size_c);
  memcpy(dc_backup, dc, sizeof(float16_t) * size_c);

  paddle::lite::operators::ActivationParam act_param;
  paddle::lite_api::ActivationType act =
      paddle::lite_api::ActivationType::kIndentity;
  if (flag_act == 1) {
    act = paddle::lite_api::ActivationType::kRelu;
  } else if (flag_act == 2) {
    act = paddle::lite_api::ActivationType::kRelu6;
    act_param.threshold = alpha;
  } else if (flag_act == 4) {
    act = paddle::lite_api::ActivationType::kLeakyRelu;
    act_param.Leaky_relu_alpha = alpha;
  }
  act_param.active_type = act;

  LOG(INFO) << "sgemm M: " << m << ", N: " << n
            << ", transA: " << (tra ? "true" : "false")
            << ", flag_act: " << (flag_act)
            << ", bias: " << (has_bias ? "true" : "false");
  if (FLAGS_check_result) {
    basic_gemv(m,
               n,
               da,
               db,
               dbias,
               dc_basic,
               static_cast<float16_t>(1.f),
               static_cast<float16_t>(0.f),
               tra,
               has_bias,
               flag_act,
               alpha);
  }
  Timer t0;
#ifdef LITE_WITH_ARM
  //! compute
  double ops = 2.0 * m * n;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  /// warmup
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::fp16::gemv_fp16(
        da, db, dc, tra, m, n, 0.f, has_bias, dbias, flag_act, act_param, &ctx);
  }

  for (int i = 0; i < FLAGS_repeats; ++i) {
    if (i == FLAGS_repeats - 1) {
      memcpy(dc, dc_backup, sizeof(float16_t) * m);
    }
    t0.Start();
    paddle::lite::arm::math::fp16::gemv_fp16(
        da, db, dc, tra, m, n, 0.f, has_bias, dbias, flag_act, act_param, &ctx);
    t0.Stop();
  }
  LOG(INFO) << "M: " << m << ", N: " << n << ", power_mode: " << cls
            << ", threads: " << ths << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t0.LapTimes().Avg()
            << " ms, min time: " << t0.LapTimes().Min()
            << " ms, mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg()
            << " GOPs, max GOPs: " << ops * 1e-6f / t0.LapTimes().Min()
            << " GOPs";
  if (FLAGS_check_result) {
    double max_ratio = 0;
    double max_diff = 0;
    auto basic_ptr = tc_basic.data<float16_t>();
    auto saber_ptr = tc.data<float16_t>();
    Tensor tdiff;
    tdiff.Resize(tc_basic.dims());
    tdiff.set_precision(PRECISION(kFP16));
    auto ptr = tdiff.mutable_data<float16_t>();
    data_diff(basic_ptr, saber_ptr, ptr, tc_basic.numel(), max_ratio, max_diff);
    print_diff_info(max_diff, max_ratio);
    int64_t size = tc_basic.numel();
    int64_t width = tc_basic.dims()[3];
    for (int i = 0; i < size; i++) {
      if (fabs(basic_ptr[i] - saber_ptr[i]) > 1e-1f &&
          fabs(basic_ptr[i] - saber_ptr[i]) /
                  (fmax(fabs(basic_ptr[i]), fabs(saber_ptr[i]))) >
              0.05) {
        print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
        LOG(FATAL) << "fp16 gemm M: " << m << ", N: " << n
                   << ", bias: " << (has_bias ? "true" : "false")
                   << ", flag_act: " << (flag_act)
                   << ", trans A: " << (tra ? "true" : "false")
                   << ", threads: " << ths << ", power_mode: " << cls
                   << ", i: " << i << " failed!!\n";
      }
    }
  }
#endif
  return true;
}

TEST(TestLiteGemvFP16, gemv_fp16) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic sgemm test";
    for (auto& m : {3, 8, 32, 397}) {
      for (auto& n : {3, 13, 141, 512, 789}) {
        for (auto& tra : {false, true}) {
          for (auto& has_bias : {false, true}) {
            for (auto& flag_act : {0, 1}) {
              for (auto& th : {1, 2, 4}) {
                float six = 6.f;
                float alpha = 8.88f;
                auto flag = test_sgemv_fp16(tra,
                                            m,
                                            n,
                                            has_bias,
                                            flag_act,
                                            FLAGS_power_mode,
                                            th,
                                            six,
                                            alpha);
                if (flag) {
                  VLOG(4) << "test m = " << m << ", n=" << n
                          << ", bias: " << (has_bias ? "true" : "false")
                          << ",  flag_act: " << (flag_act)
                          << ", trans A: " << (tra ? "true" : "false")
                          << " passed\n";
                } else {
                  LOG(FATAL) << "test m = " << m << ", n=" << n
                             << ", bias: " << (has_bias ? "true" : "false")
                             << ",  flag_act: " << (flag_act)
                             << ", trans A: " << (tra ? "true" : "false")
                             << " failed\n";
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(TestGemvFP16Custom, gemv_fp16_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_sgemv_fp16(FLAGS_traA,
                              FLAGS_M,
                              FLAGS_N,
                              FLAGS_flag_bias,
                              FLAGS_flag_act,
                              FLAGS_power_mode,
                              FLAGS_threads,
                              FLAGS_clipped_coef,
                              FLAGS_leakey_relu_alpha);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", trans A: " << FLAGS_traA << ", bias: " << FLAGS_flag_bias
               << ", act: " << FLAGS_flag_act << " failed!!";
  }
  LOG(INFO) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
            << ", trans A: " << FLAGS_traA << ", bias: " << FLAGS_flag_bias
            << ", act: " << FLAGS_flag_act << " passed!!";
}
