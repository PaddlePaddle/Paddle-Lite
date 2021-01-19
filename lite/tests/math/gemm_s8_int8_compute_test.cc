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
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;
typedef paddle::lite::operators::ActivationParam ActivationParam;

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
DEFINE_int32(K, 512, "gemv: K");

DEFINE_bool(traA, false, "gemv: A transpose");
DEFINE_bool(traB, false, "gemv: B transpose");

DEFINE_int32(flag_act, 0, "do act");
DEFINE_bool(flag_bias, false, "with bias");
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_double(clipped_coef, 6.0, "clipped relu coef");

bool test_gemm_int8(bool tra,
                    bool trb,
                    int m,
                    int n,
                    int k,
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

  ta.Resize({m, k});
  tb.Resize({k, n});
  tc_int8.Resize({m, n});
  tc_fp32.Resize({m, n});
  tc_basic_int8.Resize({m, n});
  tc_basic_fp32.Resize({m, n});
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
  std::vector<float> scale_c = {k / 127.f};
  std::vector<float> scale_merge_fp32(static_cast<size_t>(m));
  std::vector<float> scale_merge_int8(static_cast<size_t>(m));
  for (int j = 0; j < m; ++j) {
    scale_merge_fp32[j] = scale_a[j] * scale_b[0];
    scale_merge_int8[j] = scale_merge_fp32[j] / scale_c[0];
  }

  VLOG(4) << "gemm_int8 M: " << m << ", N: " << n << ", K: " << k
          << ", transA: " << (tra ? "true" : "false")
          << ", transB: " << (trb ? "true" : "false") << ", act: " << flag_act
          << ", bias: " << (has_bias ? "true" : "false");
#ifdef LITE_WITH_ARM
  int lda = tra ? m : k;
  int ldb = trb ? k : n;
  int ldc = n;
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

  bool has_relu = false;
  ActivationParam act;
  switch (flag_act) {
    case 0:
      has_relu = false;
      act.has_active = has_relu;
      break;
    case 1:
      has_relu = true;
    case 2:
    case 3:
      act.has_active = has_relu;
      act.active_type = (paddle::lite_api::ActivationType)flag_act;
      break;
    default:
      has_relu = true;
      act.has_active = has_relu;
      act.active_type = (paddle::lite_api::ActivationType)1;
  }

  if (FLAGS_check_result) {
    Tensor ta_fp32;
    Tensor tb_fp32;
    ta_fp32.Resize({m, k});
    ta_fp32.set_precision(PRECISION(kFloat));
    tb_fp32.Resize({k, n});
    tb_fp32.set_precision(PRECISION(kFloat));

    auto da_fp32 = ta_fp32.mutable_data<float>();
    auto db_fp32 = tb_fp32.mutable_data<float>();

    paddle::lite::arm::math::int8_to_fp32(
        da, da_fp32, scale_a.data(), 1, 1, ta.numel());
    paddle::lite::arm::math::int8_to_fp32(
        db, db_fp32, scale_b.data(), 1, 1, tb.numel());
    basic_gemm(tra,
               trb,
               m,
               n,
               k,
               1.f,
               da_fp32,
               lda,
               db_fp32,
               ldb,
               0.f,
               dc_basic_fp32,
               ldc,
               dbias,
               has_bias,
               has_relu);
    paddle::lite::arm::math::fp32_to_int8(dc_basic_fp32,
                                          dc_basic_int8,
                                          scale_c.data(),
                                          1,
                                          1,
                                          tc_basic_fp32.numel());
  }
  Timer t0;
  //! compute
  double ops = 2.0 * m * n * k;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  /// warmup
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::gemm_s8<float>(tra,
                                            trb,
                                            m,
                                            n,
                                            k,
                                            da,
                                            db,
                                            dc_fp32,
                                            dbias,
                                            has_bias,
                                            scale_merge_fp32.data(),
                                            act,
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
    paddle::lite::arm::math::gemm_s8<int8_t>(tra,
                                             trb,
                                             m,
                                             n,
                                             k,
                                             da,
                                             db,
                                             dc_int8,
                                             dbias_int8,
                                             has_bias,
                                             scale_merge_int8.data(),
                                             act,
                                             &ctx);
    t0.Stop();
  }
  VLOG(4) << "gemm_int8_int8 output: M: " << m << ", N: " << n << ", K: " << k
          << ", power_mode: " << cls << ", threads: " << ths
          << ", GOPS: " << ops * 1e-9f
          << " GOPS, avg time: " << t0.LapTimes().Avg()
          << " ms, min time: " << t0.LapTimes().Min()
          << " ms, mean GOPs: " << ops * 1e-6f / t0.LapTimes().Avg()
          << " GOPs, max GOPs: " << ops * 1e-6f / t0.LapTimes().Min()
          << " GOPs";

  /// fp32 output compute
  t0.Reset();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    paddle::lite::arm::math::gemm_s8<float>(tra,
                                            trb,
                                            m,
                                            n,
                                            k,
                                            da,
                                            db,
                                            dc_fp32,
                                            dbias,
                                            has_bias,
                                            scale_merge_fp32.data(),
                                            act,
                                            &ctx);
    t0.Stop();
  }
  VLOG(4) << "gemm_int8_fp32 output: M: " << m << ", N: " << n << ", K: " << k
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
    VLOG(4) << "fp32 compare result, max diff: " << max_diff
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
    VLOG(4) << "int8 compare result, max diff: " << max_diff
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
        LOG(INFO) << "int8 basic result";
        print_tensor(tc_basic_int8);
        LOG(INFO) << "int8 lite result";
        print_tensor(tc_int8);
        LOG(INFO) << "int8 diff tensor";
        print_tensor(tdiff);
        return false;
      }
    }
  }
#endif
  return true;
}

TEST(TestLiteGemmInt8, gemm_int8) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    VLOG(4) << "run basic sgemm test";
    for (auto& m : {1, 3, 8, 32, 33, 34, 35, 38, 41, 397}) {
      for (auto& n : {1, 3, 13, 141, 512, 789}) {
        for (auto& k : {1, 3, 8, 59, 60, 61, 62, 66, 67, 71}) {
          for (auto& tra : {false, true}) {
            for (auto& trb : {false, true}) {
              for (auto& has_bias : {false, true}) {
                for (auto& relu_type : {0, 1}) {
                  for (auto& th : {1}) {
                    auto flag = true;
                    if (m == 1 || n == 1) {
                      flag = test_gemm_int8(false,
                                            false,
                                            m,
                                            n,
                                            k,
                                            has_bias,
                                            relu_type,
                                            FLAGS_power_mode,
                                            th);
                    } else {
                      flag = test_gemm_int8(tra,
                                            trb,
                                            m,
                                            n,
                                            k,
                                            has_bias,
                                            relu_type,
                                            FLAGS_power_mode,
                                            th);
                    }
                    if (flag) {
                      VLOG(4) << "test m = " << m << ", n=" << n << ", k=" << k
                              << ", bias: " << (has_bias ? "true" : "false")
                              << ", relu: " << relu_type
                              << ", trans A: " << (tra ? "true" : "false")
                              << ", trans B: " << (trb ? "true" : "false")
                              << " passed\n";
                    } else {
                      LOG(FATAL) << "test m = " << m << ", n=" << n
                                 << ", k=" << k
                                 << ", bias: " << (has_bias ? "true" : "false")
                                 << ", relu: " << relu_type
                                 << ", trans A: " << (tra ? "true" : "false")
                                 << ", trans B: " << (trb ? "true" : "false")
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
  }
}

TEST(TestGemmInt8Custom, gemm_int8_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_gemm_int8(FLAGS_traA,
                             FLAGS_traB,
                             FLAGS_M,
                             FLAGS_N,
                             FLAGS_K,
                             FLAGS_flag_bias,
                             FLAGS_flag_act,
                             FLAGS_power_mode,
                             FLAGS_threads,
                             FLAGS_clipped_coef,
                             FLAGS_leakey_relu_alpha);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", k=" << FLAGS_K << ", trans A: " << FLAGS_traA
               << ", trans B: " << FLAGS_traB << ", bias: " << FLAGS_flag_bias
               << ", act: " << FLAGS_flag_act << " failed!!";
  }
  VLOG(4) << "test m = " << FLAGS_M << ", n=" << FLAGS_N << ", k=" << FLAGS_K
          << ", trans A: " << FLAGS_traA << ", trans B: " << FLAGS_traB
          << ", bias: " << FLAGS_flag_bias << ", act: " << FLAGS_flag_act
          << " passed!!";
}
