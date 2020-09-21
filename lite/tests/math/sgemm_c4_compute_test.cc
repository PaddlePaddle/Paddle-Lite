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

DEFINE_int32(M, 512, "gemm_c4: M");
DEFINE_int32(N, 512, "gemm_c4: N");
DEFINE_int32(K, 512, "gemm_c4: K");

DEFINE_bool(flag_relu, false, "do relu");
DEFINE_bool(flag_bias, false, "with bias");

bool test_sgemm_c4(
    int m, int n, int k, bool has_bias, bool has_relu, int cls, int ths) {
  int m_round = (m + 3) / 4 * 4;
  int k_round = (k + 3) / 4 * 4;
  int size_a = m * k;
  int size_b = n * k;
  int size_a_c4 = m_round * k_round;
  int size_b_c4 = k_round * n;

  Tensor ta;
  Tensor tb;
  Tensor ta_c4;
  Tensor tb_c4;
  Tensor tc;
  Tensor tc_basic;
  Tensor tc_backup;
  Tensor tbias;

  ta.Resize({size_a});
  tb.Resize({size_b});
  ta_c4.Resize({size_a_c4});
  tb_c4.Resize({size_b_c4});
  tc.Resize({m_round * n});
  tc_basic.Resize({m_round * n});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kFloat));
  tb.set_precision(PRECISION(kFloat));
  ta_c4.set_precision(PRECISION(kFloat));
  tb_c4.set_precision(PRECISION(kFloat));
  tc.set_precision(PRECISION(kFloat));
  tc_basic.set_precision(PRECISION(kFloat));
  tbias.set_precision(PRECISION(kFloat));

  fill_tensor_rand(ta, -1.f, 1.f);
  fill_tensor_rand(tb, -1.f, 1.f);
  fill_tensor_rand(tbias, -1.f, 1.f);
  fill_tensor_rand(tc, -1.f, 1.f);

  auto da = ta.mutable_data<float>();
  auto db = tb.mutable_data<float>();
  auto da_c4 = ta_c4.mutable_data<float>();
  auto db_c4 = tb_c4.mutable_data<float>();
  auto dc_basic = tc_basic.mutable_data<float>();
  auto dbias = tbias.mutable_data<float>();
  memset(reinterpret_cast<char*>(dc_basic), 0, tc_basic.numel());

  // trans A, B to c4
  basic_trans_mat_to_c4(da, da_c4, k, m, k, true);
  basic_trans_mat_to_c4(db, db_c4, n, k, n, false);

  VLOG(4) << "sgemm_c4 M: " << m << ", N: " << n << ", K: " << k
          << ", relu: " << (has_relu ? "true" : "false")
          << ", bias: " << (has_bias ? "true" : "false");

  if (FLAGS_check_result) {
    basic_gemm_c4(false,
                  false,
                  m,
                  n,
                  k,
                  1.f,
                  da,
                  k,
                  db,
                  n,
                  0.f,
                  dc_basic,
                  n,
                  dbias,
                  has_bias,
                  has_relu);
  }
  Timer t0;
#ifdef LITE_WITH_ARM
  //! compute
  double ops = 2.0 * m_round * n * k_round;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  auto dc = tc.mutable_data<float>();
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::sgemm_prepack_c4(
        m, n, k, da_c4, db_c4, dc, dbias, has_bias, has_relu, &ctx);
  }

  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    paddle::lite::arm::math::sgemm_prepack_c4(
        m, n, k, da_c4, db_c4, dc, dbias, has_bias, has_relu, &ctx);
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
      LOG(INFO) << "a_c4: ";
      print_tensor(ta_c4);
      LOG(INFO) << "b: ";
      print_tensor(tb);
      LOG(INFO) << "b_c4: ";
      print_tensor(tb_c4);
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
bool test_sgemm_c8(
    int m, int n, int k, bool has_bias, bool has_relu, int cls, int ths) {
  int m_round = (m + 7) / 8 * 8;
  int k_round = (k + 7) / 8 * 8;
  int size_a = m * k;
  int size_b = n * k;
  int size_a_c4 = m_round * k_round;
  int size_b_c8 = k_round * n;

  Tensor ta;
  Tensor tb;
  Tensor ta_c4;
  Tensor tb_c8;
  Tensor tc;
  Tensor tc_basic;
  Tensor tc_backup;
  Tensor tbias;

  ta.Resize({size_a});
  tb.Resize({size_b});
  ta_c4.Resize({size_a_c4});
  tb_c8.Resize({size_b_c8});
  tc.Resize({m_round * n});
  tc_basic.Resize({m_round * n});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kInt16));
  tb.set_precision(PRECISION(kInt16));
  ta_c4.set_precision(PRECISION(kInt16));
  tb_c8.set_precision(PRECISION(kInt16));
  tc.set_precision(PRECISION(kInt32));
  tc_basic.set_precision(PRECISION(kInt32));
  tbias.set_precision(PRECISION(kInt32));

  fill_tensor_rand(ta);
  fill_tensor_rand(tb);
  fill_tensor_rand(tbias);
  fill_tensor_rand(tc);

  auto da = ta.mutable_data<int16_t>();
  auto db = tb.mutable_data<int16_t>();
  auto da_c4 = ta_c4.mutable_data<int16_t>();
  auto db_c8 = tb_c8.mutable_data<int16_t>();
  auto dc_basic = tc_basic.mutable_data<int32_t>();
  auto dbias = tbias.mutable_data<int32_t>();

  // trans A, B to c4
  basic_trans_mat_to_c8(da, da_c4, k, m, k, true);
  basic_trans_mat_to_c8(db, db_c8, n, k, n, false);

  LOG(INFO) << "sgemm_c8 M: " << m << ", N: " << n << ", K: " << k
            << ", relu: " << (has_relu ? "true" : "false")
            << ", bias: " << (has_bias ? "true" : "false");

  if (FLAGS_check_result) {
    basic_gemm_c8(false,
                  false,
                  m,
                  n,
                  k,
                  1,
                  da,
                  k,
                  db,
                  n,
                  0,
                  dc_basic,
                  n,
                  dbias,
                  false,
                  false);
  }
  Timer t0;
  LOG(INFO) << "basic test end";
#ifdef LITE_WITH_ARM
  //! compute
  double ops = 2.0 * m_round * n * k_round;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  auto dc = tc.mutable_data<int32_t>();
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::sgemm_prepack_c8_int16_small(
        m, n, k, da_c4, db_c8, dc, &ctx);
  }
  LOG(INFO) << "basic test end";

  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    paddle::lite::arm::math::sgemm_prepack_c8_int16_small(
        m, n, k, da_c4, db_c8, dc, &ctx);
    t0.Stop();
  }
  LOG(INFO) << "basic test end";
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
      tdiff.set_precision(PRECISION(kInt32));
      tdiff.Resize(tc.dims());
      tensor_diff(tc_basic, tc, tdiff);
      LOG(INFO) << "a: ";
      print_tensor(ta);
      LOG(INFO) << "a_c8: ";
      print_tensor(ta_c4);
      LOG(INFO) << "b: ";
      print_tensor(tb);
      LOG(INFO) << "b_c8: ";
      print_tensor(tb_c8);
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

TEST(TestSgemmC4, test_func_sgemm_c4_prepacked) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic sgemm_c4 test";
    for (auto& m : {1, 3, 8, 32, 397, 32, 64, 77}) {
      for (auto& n : {1, 2, 3, 4, 13, 141, 789, 1}) {
        for (auto& k : {1, 3, 8, 59, 234, 19}) {
          for (auto& has_bias : {false}) {
            for (auto& has_relu : {false}) {
              for (auto& th : {1, 2, 4}) {
                auto flag = test_sgemm_c4(
                    m, n, k, has_bias, has_relu, FLAGS_power_mode, th);
                if (flag) {
                  VLOG(4) << "test m = " << m << ", n=" << n << ", k=" << k
                          << ", bias: " << (has_bias ? "true" : "false")
                          << ", relu: " << (has_relu ? "true" : "false")
                          << " passed\n";
                } else {
                  LOG(FATAL) << "test m = " << m << ", n=" << n << ", k=" << k
                             << ", bias: " << (has_bias ? "true" : "false")
                             << ", relu: " << (has_relu ? "true" : "false")
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
TEST(TestSgemmC8, test_func_sgemm_c8_prepacked) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic sgemm_c4 test";
    for (auto& m : {1, 3, 8, 32, 397, 32, 64, 77}) {
      for (auto& n : {1, 2, 3, 4, 13, 141, 789, 1}) {
        for (auto& k : {1, 3, 8, 59, 234, 19}) {
          for (auto& has_bias : {false}) {
            for (auto& has_relu : {false}) {
              for (auto& th : {1}) {
                auto flag = test_sgemm_c8(
                    m, n, k, has_bias, has_relu, FLAGS_power_mode, th);
                if (flag) {
                  VLOG(4) << "test m = " << m << ", n=" << n << ", k=" << k
                          << ", bias: " << (has_bias ? "true" : "false")
                          << ", relu: " << (has_relu ? "true" : "false")
                          << " passed\n";
                } else {
                  LOG(FATAL) << "test m = " << m << ", n=" << n << ", k=" << k
                             << ", bias: " << (has_bias ? "true" : "false")
                             << ", relu: " << (has_relu ? "true" : "false")
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

TEST(TestSgemmCnCustom, test_func_sgemm_cn_prepacked_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_sgemm_c4(FLAGS_M,
                            FLAGS_N,
                            FLAGS_K,
                            FLAGS_flag_bias,
                            FLAGS_flag_relu,
                            FLAGS_power_mode,
                            FLAGS_threads);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", k=" << FLAGS_K << ", bias: " << FLAGS_flag_bias
               << ", relu: " << FLAGS_flag_relu << " failed!!";
  }
  flag = test_sgemm_c8(FLAGS_M,
                       FLAGS_N,
                       FLAGS_K,
                       FLAGS_flag_bias,
                       FLAGS_flag_relu,
                       FLAGS_power_mode,
                       FLAGS_threads);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", k=" << FLAGS_K << ", bias: " << FLAGS_flag_bias
               << ", relu: " << FLAGS_flag_relu << " failed!!";
  }
  LOG(INFO) << "test m = " << FLAGS_M << ", n=" << FLAGS_N << ", k=" << FLAGS_K
            << ", bias: " << FLAGS_flag_bias << ", relu: " << FLAGS_flag_relu
            << " passed!!";
}
