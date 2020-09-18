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
typedef paddle::lite::operators::ActivationParam ActivationParam;
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

#ifdef LITE_WITH_ARM
// sgemm_test wiil not be operated except that it's
// on arm backend.
DEFINE_bool(basic_test, true, "do all tests");
#else
DEFINE_bool(basic_test, false, "do all tests");
#endif

DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(M, 512, "gemm: M");
DEFINE_int32(N, 512, "gemm: N");
DEFINE_int32(K, 512, "gemm: K");

DEFINE_bool(traA, false, "gemm: A transpose");
DEFINE_bool(traB, false, "gemm: B transpose");

DEFINE_int32(offset_a, 0, "A offset");
DEFINE_int32(offset_b, 0, "B offset");
DEFINE_int32(offset_c, 0, "C offset");

DEFINE_double(alpha, 1.0, "alpha");
DEFINE_double(beta, 0.0, "beta");

DEFINE_bool(flag_relu, false, "do relu");
DEFINE_bool(flag_bias, false, "with bias");

bool test_sgemm(bool tra,
                bool trb,
                int m,
                int n,
                int k,
                int lda,
                int ldb,
                int ldc,
                float alpha,
                float beta,
                bool has_bias,
                bool has_relu,
                int cls,
                int ths) {
  int size_a = tra ? k * lda : m * lda;
  int size_b = trb ? n * ldb : k * ldb;

  Tensor ta;
  Tensor tb;
  Tensor tc;
  Tensor tc_basic;
  Tensor tc_backup;
  Tensor tbias;

  ta.Resize({size_a});
  tb.Resize({size_b});
  tc.Resize({m * ldc});
  tc_basic.Resize({m * ldc});
  tc_backup.Resize({m * ldc});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kFloat));
  tb.set_precision(PRECISION(kFloat));
  tc.set_precision(PRECISION(kFloat));
  tc_basic.set_precision(PRECISION(kFloat));
  tc_backup.set_precision(PRECISION(kFloat));
  tbias.set_precision(PRECISION(kFloat));

  fill_tensor_rand(ta, -1.f, 1.f);
  fill_tensor_rand(tb, -1.f, 1.f);
  fill_tensor_rand(tbias, -1.f, 1.f);
  fill_tensor_rand(tc, -1.f, 1.f);

  auto da = ta.mutable_data<float>();
  auto db = tb.mutable_data<float>();
  auto dc = tc.mutable_data<float>();
  auto dc_basic = tc_basic.mutable_data<float>();
  auto dc_backup = tc_backup.mutable_data<float>();
  auto dbias = tbias.mutable_data<float>();

  memcpy(dc_basic, dc, sizeof(float) * m * ldc);
  memcpy(dc_backup, dc, sizeof(float) * m * ldc);

  VLOG(4) << "sgemm M: " << m << ", N: " << n << ", K: " << k
          << ", strides, lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
          << ", alpha: " << alpha << ", beta: " << beta
          << ", transA: " << (tra ? "true" : "false")
          << ", transB: " << (trb ? "true" : "false")
          << ", relu: " << (has_relu ? "true" : "false")
          << ", bias: " << (has_bias ? "true" : "false");
  if (FLAGS_check_result) {
    basic_gemm(tra,
               trb,
               m,
               n,
               k,
               alpha,
               da,
               lda,
               db,
               ldb,
               beta,
               dc_basic,
               ldc,
               dbias,
               has_bias,
               has_relu);
  }
  Timer t0;
  ActivationParam act_param;
  if (has_relu) {
    act_param.has_active = true;
    act_param.active_type =
        (paddle::lite_api::ActivationType)1;  // 2-relu6 4-leakyrelu
  }
#ifdef LITE_WITH_ARM
  //! compute
  double ops = 2.0 * m * n * k;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  //! prepack
  Tensor tpackedA;
  int hblock = paddle::lite::arm::math::get_hblock(&ctx);
  int round_up_a = ((hblock + m - 1) / hblock) * hblock;
  tpackedA.Resize({round_up_a * k});
  paddle::lite::arm::math::prepackA(
      tpackedA.mutable_data<float>(), da, alpha, lda, 0, m, 0, k, tra, &ctx);
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::sgemm_prepack(trb,
                                           m,
                                           n,
                                           k,
                                           tpackedA.data<float>(),
                                           db,
                                           ldb,
                                           beta,
                                           dc,
                                           ldc,
                                           dbias,
                                           has_bias,
                                           act_param,
                                           &ctx);
  }

  for (int i = 0; i < FLAGS_repeats; ++i) {
    if (i == FLAGS_repeats - 1) {
      memcpy(dc, dc_backup, sizeof(float) * m * ldc);
    }
    t0.Start();
    paddle::lite::arm::math::sgemm_prepack(trb,
                                           m,
                                           n,
                                           k,
                                           tpackedA.data<float>(),
                                           db,
                                           ldb,
                                           beta,
                                           dc,
                                           ldc,
                                           dbias,
                                           has_bias,
                                           act_param,
                                           &ctx);
    t0.Stop();
  }
  LOG(INFO) << "M: " << m << ", N: " << n << ", K: " << k
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
    tensor_cmp_host(tc_basic, tc, max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio;
    if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
      Tensor tdiff;
      tdiff.set_precision(PRECISION(kFloat));
      tdiff.Resize(tc.dims());
      tensor_diff(tc_basic, tc, tdiff);
      LOG(INFO) << "a: ";
      print_tensor(ta);
      LOG(INFO) << "b: ";
      print_tensor(tb);
      LOG(INFO) << "c: ";
      print_tensor(tc_backup);
      LOG(INFO) << "basic result: ";
      print_tensor(tc_basic);
      LOG(INFO) << "lite result: ";
      print_tensor(tc);
      LOG(INFO) << "diff result: ";
      print_tensor(tdiff);
      return false;
    }
  }
#endif
  return true;
}

TEST(TestSgemm, test_func_sgemm_prepacked) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic sgemm test";
    for (auto& m : {1, 3, 8, 32, 397}) {
      for (auto& n : {1, 3, 13, 141, 512, 789}) {
        for (auto& k : {1, 3, 8, 59, 234}) {
          for (auto& tra : {false, true}) {
            for (auto& trb : {false, true}) {
              for (auto& alpha : {1.f, 0.5f}) {
                for (auto& beta : {0.f, 0.5f}) {
                  for (auto& offset : {0, 10}) {
                    for (auto& has_bias : {false, true}) {
                      for (auto& has_relu : {false, true}) {
                        for (auto& th : {1, 2, 4}) {
                          int lda = k + offset;
                          if (tra) {
                            lda = m + offset;
                          }
                          int ldb = n + offset;
                          if (trb) {
                            ldb = k + offset;
                          }
                          int ldc = n + offset;
                          auto flag = test_sgemm(tra,
                                                 trb,
                                                 m,
                                                 n,
                                                 k,
                                                 lda,
                                                 ldb,
                                                 ldc,
                                                 alpha,
                                                 beta,
                                                 has_bias,
                                                 has_relu,
                                                 FLAGS_power_mode,
                                                 th);
                          if (flag) {
                            VLOG(4)
                                << "test m = " << m << ", n=" << n
                                << ", k=" << k
                                << ", bias: " << (has_bias ? "true" : "false")
                                << ", relu: " << (has_relu ? "true" : "false")
                                << ", trans A: " << (tra ? "true" : "false")
                                << ", trans B: " << (trb ? "true" : "false")
                                << " passed\n";
                          } else {
                            LOG(FATAL)
                                << "test m = " << m << ", n=" << n
                                << ", k=" << k
                                << ", bias: " << (has_bias ? "true" : "false")
                                << ", relu: " << (has_relu ? "true" : "false")
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
    }
  }
}

TEST(TestSgemmCustom, test_func_sgemm_prepacked_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  int lda = FLAGS_K + FLAGS_offset_a;
  if (FLAGS_traA) {
    lda = FLAGS_M + FLAGS_offset_a;
  }
  int ldb = FLAGS_N + FLAGS_offset_b;
  if (FLAGS_traB) {
    ldb = FLAGS_K + FLAGS_offset_b;
  }
  int ldc = FLAGS_N + FLAGS_offset_c;
  auto flag = test_sgemm(FLAGS_traA,
                         FLAGS_traB,
                         FLAGS_M,
                         FLAGS_N,
                         FLAGS_K,
                         lda,
                         ldb,
                         ldc,
                         FLAGS_alpha,
                         FLAGS_beta,
                         FLAGS_flag_bias,
                         FLAGS_flag_relu,
                         FLAGS_power_mode,
                         FLAGS_threads);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", k=" << FLAGS_K << ", trans A: " << FLAGS_traA
               << ", trans B: " << FLAGS_traB << ", bias: " << FLAGS_flag_bias
               << ", relu: " << FLAGS_flag_relu << " failed!!";
  }
  LOG(INFO) << "test m = " << FLAGS_M << ", n=" << FLAGS_N << ", k=" << FLAGS_K
            << ", trans A: " << FLAGS_traA << ", trans B: " << FLAGS_traB
            << ", bias: " << FLAGS_flag_bias << ", relu: " << FLAGS_flag_relu
            << " passed!!";
}
