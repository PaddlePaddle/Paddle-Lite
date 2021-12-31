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

DEFINE_bool(basic_test, true, "do all tests");
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

bool test_sgemm_fp16(bool tra,
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
  int size_c = m * ldc;

  Tensor ta;
  Tensor ta_fp32;
  Tensor tb;
  Tensor tb_fp32;
  Tensor tc;
  Tensor tc_basic;
  Tensor tc_basic_fp32;
  Tensor tc_backup;
  Tensor tbias;
  Tensor tbias_fp32;

  ta.Resize({size_a});
  ta_fp32.Resize({size_a});
  tb.Resize({size_b});
  tb_fp32.Resize({size_b});
  tc.Resize({size_c});
  tc_basic.Resize({size_c});
  tc_basic_fp32.Resize({size_c});
  tc_backup.Resize({size_c});
  tbias.Resize({m});
  tbias_fp32.Resize({m});

  ta.set_precision(PRECISION(kFP16));
  tb.set_precision(PRECISION(kFP16));
  tc.set_precision(PRECISION(kFP16));
  tc_basic.set_precision(PRECISION(kFP16));
  tc_backup.set_precision(PRECISION(kFP16));
  tbias.set_precision(PRECISION(kFP16));
  ta_fp32.set_precision(PRECISION(kFloat));
  tb_fp32.set_precision(PRECISION(kFloat));
  tbias_fp32.set_precision(PRECISION(kFloat));
  tc_basic_fp32.set_precision(PRECISION(kFloat));

  auto da = ta.mutable_data<float16_t>();
  auto db = tb.mutable_data<float16_t>();
  auto dc = tc.mutable_data<float16_t>();
  auto dc_basic = tc_basic.mutable_data<float16_t>();
  auto dc_backup = tc_backup.mutable_data<float16_t>();
  auto dbias = tbias.mutable_data<float16_t>();
  auto da_fp32 = ta_fp32.mutable_data<float>();
  auto db_fp32 = tb_fp32.mutable_data<float>();
  auto dbias_fp32 = tbias_fp32.mutable_data<float>();
  auto dc_basic_fp32 = tc_basic_fp32.mutable_data<float>();

  fill_data_rand<float16_t>(da, -1.f, 1.f, size_a);
  // fill_data_const<float16_t>(da, 1.f, size_a);
  fp16_to_float(da, da_fp32, size_a);

  fill_data_rand<float16_t>(db, -1.f, 1.f, size_b);
  // fill_data_const<float16_t>(db, 1.f, size_b);
  fp16_to_float(db, db_fp32, size_b);

  fill_data_rand<float16_t>(dbias, -1.f, 1.f, m);
  // fill_data_const<float16_t>(dbias, -1.f, m);
  fp16_to_float(dbias, dbias_fp32, m);

  // fill_data_rand<float16_t>(dc, -1.f, 1.f, size_c);
  fill_data_const<float16_t>(dc, 1.f, size_c);
  fp16_to_float(dc, dc_basic_fp32, size_c);

  memcpy(dc_basic, dc, sizeof(float16_t) * size_c);
  memcpy(dc_backup, dc, sizeof(float16_t) * size_c);

  LOG(INFO) << "sgemm M: " << m << ", N: " << n << ", K: " << k
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
               static_cast<float16_t>(alpha),
               da,
               lda,
               db,
               ldb,
               static_cast<float16_t>(beta),
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
  int hblock = paddle::lite::arm::math::fp16::get_hblock_fp16(&ctx);
  int round_up_a = ((hblock + m - 1) / hblock) * hblock;
  tpackedA.Resize({round_up_a * k});
  paddle::lite::arm::math::fp16::prepackA_fp16(
      tpackedA.mutable_data<float16_t>(),
      da,
      alpha,
      lda,
      0,
      m,
      0,
      k,
      tra,
      &ctx);
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::fp16::gemm_prepack_fp16(
        trb,
        m,
        n,
        k,
        tpackedA.data<float16_t>(),
        db,
        ldb,
        static_cast<float16_t>(beta),
        dc,
        ldc,
        dbias,
        has_bias,
        act_param,
        &ctx);
  }

  for (int i = 0; i < FLAGS_repeats; ++i) {
    if (i == FLAGS_repeats - 1) {
      memcpy(dc, dc_backup, sizeof(float16_t) * m * ldc);
    }
    t0.Start();
    paddle::lite::arm::math::fp16::gemm_prepack_fp16(
        trb,
        m,
        n,
        k,
        tpackedA.data<float16_t>(),
        db,
        ldb,
        static_cast<float16_t>(beta),
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
        LOG(FATAL) << "fp16 gemm M: " << m << ", N: " << n << ", K: " << k
                   << ", bias: " << (has_bias ? "true" : "false")
                   << ", relu: " << (has_relu ? "true" : "false")
                   << ", trans A: " << (tra ? "true" : "false")
                   << ", trans B: " << (trb ? "true" : "false")
                   << ", threads: " << ths << ", power_mode: " << cls
                   << ", i: " << i << " failed!!\n";
      }
    }
  }
#endif
  return true;
}

TEST(TestSgemm, test_func_sgemm_prepacked_fp16) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    for (auto& m : {3, 8, 32, 397}) {
      for (auto& n : {3, 13, 141, 512, 789}) {
        for (auto& k : {1, 3, 8, 59, 234}) {
          for (auto& tra : {false, true}) {
            for (auto& trb : {false, true}) {
              for (auto& alpha : {1.f, 0.5f}) {
                for (auto& beta : {0.f, 0.5f}) {
                  for (auto& offset : {0}) {
                    for (auto& has_bias : {false}) {  //, true
                      for (auto& has_relu : {false, true}) {
                        for (auto& th : {1}) {  //, 2, 4
                                                //   offset = 0;
                          int lda = k + offset;
                          if (tra) {
                            lda = m + offset;
                          }
                          int ldb = n + offset;
                          if (trb) {
                            ldb = k + offset;
                          }
                          int ldc = n + offset;
                          auto flag = test_sgemm_fp16(tra,
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
                            LOG(INFO)
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

TEST(TestSgemmCustom, test_func_sgemm_prepacked_fp16_custom) {
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
  auto flag = test_sgemm_fp16(FLAGS_traA,
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
