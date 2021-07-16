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
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
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
DEFINE_bool(basic_test, true, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(M, 512, "gemv: M");
DEFINE_int32(N, 512, "gemv: N");

DEFINE_bool(traA, false, "gemv: A transpose");

DEFINE_int32(flag_act, 0, "do act");
DEFINE_bool(flag_bias, false, "with bias");
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_double(clipped_coef, 6.0, "clipped relu coef");

bool test_gemv_int8(bool tra,
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
  Tensor tc_int8;
  Tensor tc_fp32;
  Tensor tc_basic_int8;
  Tensor tc_basic_fp32;
  Tensor tbias;

  ta.Resize({m, n});
  tb.Resize({n});
  tc_int8.Resize({m});
  tc_fp32.Resize({m});
  tc_basic_int8.Resize({m});
  tc_basic_fp32.Resize({m});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kInt8));
  tb.set_precision(PRECISION(kInt8));
  tc_int8.set_precision(PRECISION(kInt8));
  tc_fp32.set_precision(PRECISION(kFloat));
  tc_basic_int8.set_precision(PRECISION(kInt8));
  tc_basic_fp32.set_precision(PRECISION(kFloat));
  tbias.set_precision(PRECISION(kFloat));

  fill_tensor_rand(ta, -127, 127);
  fill_tensor_rand(tb, -127, 127);
  fill_tensor_rand(tbias, -1.f, 1.f);

  std::vector<float> scale_a(static_cast<size_t>(m), 1.f / 127);
  std::vector<float> scale_b = {1.f / 127};
  std::vector<float> scale_c = {n / 127.f};
  std::vector<float> scale_merge_fp32(static_cast<size_t>(m));
  std::vector<float> scale_merge_int8(static_cast<size_t>(m));
  for (int j = 0; j < m; ++j) {
    scale_merge_fp32[j] = scale_a[j] * scale_b[0];
    scale_merge_int8[j] = scale_merge_fp32[j] / scale_c[0];
  }

  VLOG(4) << "gemv_int8 M: " << m << ", N: " << n
          << ", transA: " << (tra ? "true" : "false") << ", act: " << flag_act
          << ", bias: " << (has_bias ? "true" : "false");
#ifdef LITE_WITH_ARM
  auto da = ta.mutable_data<int8_t>();
  auto db = tb.mutable_data<int8_t>();
  auto dc_int8 = tc_int8.mutable_data<int8_t>();
  auto dc_fp32 = tc_fp32.mutable_data<float>();
  auto dc_basic_int8 = tc_basic_int8.mutable_data<int8_t>();
  auto dc_basic_fp32 = tc_basic_fp32.mutable_data<float>();
  auto dbias = tbias.mutable_data<float>();
  // set intial input to be 0
  memset(reinterpret_cast<char*>(dc_basic_fp32),
         0,
         tc_basic_fp32.numel() * sizeof(float));
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

  if (FLAGS_check_result) {
    Tensor ta_fp32;
    Tensor tb_fp32;
    ta_fp32.Resize({m, n});
    ta_fp32.set_precision(PRECISION(kFloat));
    tb_fp32.Resize({n});
    tb_fp32.set_precision(PRECISION(kFloat));

    auto da_fp32 = ta_fp32.mutable_data<float>();
    auto db_fp32 = tb_fp32.mutable_data<float>();

    paddle::lite::arm::math::int8_to_fp32(
        da, da_fp32, scale_a.data(), 1, 1, ta.numel());
    paddle::lite::arm::math::int8_to_fp32(
        db, db_fp32, scale_b.data(), 1, 1, tb.numel());
    basic_gemv(m,
               n,
               da_fp32,
               db_fp32,
               dbias,
               dc_basic_fp32,
               1.f,
               0.f,
               false,
               has_bias,
               flag_act,
               alpha);
    paddle::lite::arm::math::fp32_to_int8(dc_basic_fp32,
                                          dc_basic_int8,
                                          scale_c.data(),
                                          1,
                                          1,
                                          tc_basic_fp32.numel());
  }
  Timer t0;
  //! compute
  double ops = 2.0 * m * n;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  /// warmup
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::gemv_int8(da,
                                       db,
                                       dc_fp32,
                                       false,
                                       m,
                                       n,
                                       scale_merge_fp32.data(),
                                       has_bias,
                                       dbias,
                                       act_param,
                                       &ctx);
  }

  /// int8 output compute
  Tensor tbias_int8;
  tbias_int8.Resize(tbias.dims());
  tbias_int8.set_precision(PRECISION(kFloat));
  auto dbias_int8 = tbias_int8.mutable_data<float>();
  for (int l = 0; l < tbias_int8.numel(); ++l) {
    dbias_int8[l] = dbias[l] / scale_c[0];
  }
  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    paddle::lite::arm::math::gemv_int8(da,
                                       db,
                                       dc_fp32,
                                       false,
                                       m,
                                       n,
                                       scale_merge_fp32.data(),
                                       has_bias,
                                       dbias,
                                       act_param,
                                       &ctx);
    t0.Stop();
  }
  LOG(INFO) << "gemv_int8_int8 output: M: " << m << ", N: " << n
            << ", power_mode: " << cls << ", threads: " << ths
            << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t0.LapTimes().Avg()
            << " ms, min time: " << t0.LapTimes().Min()
            << " ms, mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg()
            << " GOPs, max GOPs: " << ops * 1e-6f / t0.LapTimes().Min()
            << " GOPs";

  /// fp32 output compute
  if (flag_act == 2) {
    alpha = alpha / scale_c[0];
  }
  t0.Reset();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    paddle::lite::arm::math::gemv_int8(da,
                                       db,
                                       dc_int8,
                                       false,
                                       m,
                                       n,
                                       scale_merge_int8.data(),
                                       has_bias,
                                       dbias_int8,
                                       act_param,
                                       &ctx);
    t0.Stop();
  }
  LOG(INFO) << "gemm_int8_fp32 output: M: " << m << ", N: " << n
            << ", power_mode: " << cls << ", threads: " << ths
            << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t0.LapTimes().Avg()
            << " ms, min time: " << t0.LapTimes().Min()
            << " ms, mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg()
            << " GOPs, max GOPs: " << ops * 1e-6f / t0.LapTimes().Min()
            << " GOPs";

  if (FLAGS_check_result) {
    double max_ratio = 0;
    double max_diff = 0;
    /// fp32 result
    tensor_cmp_host(tc_basic_fp32, tc_fp32, max_ratio, max_diff);
    LOG(INFO) << "fp32 compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio;
    if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
      Tensor tdiff;
      tdiff.set_precision(PRECISION(kFloat));
      tdiff.Resize(tc_fp32.dims());
      tensor_diff(tc_basic_fp32, tc_fp32, tdiff);
      LOG(INFO) << "basic result: ";
      print_tensor(tc_basic_fp32);
      LOG(INFO) << "lite result: ";
      print_tensor(tc_fp32);
      LOG(INFO) << "diff result: ";
      print_tensor(tdiff);
      return false;
    }
    /// int8 result
    max_ratio = 0;
    max_diff = 0;
    tensor_cmp_host(tc_basic_int8, tc_int8, max_ratio, max_diff);
    LOG(INFO) << "int8 compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio;
    if (fabs(max_ratio) > 1e-4f) {
      Tensor tdiff;
      tdiff.Resize(tc_int8.dims());
      tdiff.set_precision(PRECISION(kInt8));
      tensor_diff(tc_basic_int8, tc_int8, tdiff);
      auto ptr = tdiff.data<int8_t>();
      auto ptr_basic_fp32 = tc_basic_fp32.data<float>();
      float count = 0;
      bool check = true;
      for (int i = 0; i < tdiff.numel(); ++i) {
        if (abs(ptr[i]) > 1) {
          check = false;
          LOG(ERROR) << "basic float data: " << ptr_basic_fp32[i]
                     << ", after scale: " << ptr_basic_fp32[i] / scale_c[0];
          break;
        }
        if (ptr[i] != 0) {
          LOG(ERROR) << "basic float data: " << ptr_basic_fp32[i]
                     << ", after scale: " << ptr_basic_fp32[i] / scale_c[0];
          count += 1;
        }
      }
      check =
          check && count < std::max(10, static_cast<int>(0.01 * tdiff.numel()));
      if (!check) {
        LOG(WARNING) << "int8 basic result";
        print_tensor(tc_basic_int8);
        LOG(WARNING) << "int8 lite result";
        print_tensor(tc_int8);
        LOG(WARNING) << "int8 diff tensor";
        print_tensor(tdiff);
        return false;
      }
    }
  }
#endif
  return true;
}

TEST(TestLiteGemvInt8, gemv_prepacked_int8) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic sgemm test";
    for (auto& m : {3, 8, 32, 397}) {
      for (auto& n : {3, 13, 141, 512, 789}) {
        for (auto& tra : {false}) {
          for (auto& has_bias : {false, true}) {
            for (auto& has_relu : {false, true}) {
              for (auto& th : {1, 2, 4}) {
                float six = 6.f;
                float alpha = 8.88f;
                auto flag = test_gemv_int8(tra,
                                           m,
                                           n,
                                           has_bias,
                                           has_relu > 0,
                                           FLAGS_power_mode,
                                           th,
                                           six,
                                           alpha);
                if (flag) {
                  VLOG(4) << "test m = " << m << ", n=" << n
                          << ", bias: " << (has_bias ? "true" : "false")
                          << ",  relu: " << (has_relu ? "true" : "false")
                          << ", trans A: " << (tra ? "true" : "false")
                          << " passed\n";
                } else {
                  LOG(FATAL) << "test m = " << m << ", n=" << n
                             << ", bias: " << (has_bias ? "true" : "false")
                             << ",  relu: " << (has_relu ? "true" : "false")
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

TEST(TestGemvInt8Custom, gemv_prepacked_int8_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_gemv_int8(FLAGS_traA,
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
