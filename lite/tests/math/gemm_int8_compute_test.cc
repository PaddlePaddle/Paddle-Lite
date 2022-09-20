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
#ifdef LITE_WITH_ARM8_SVE2
#include "lite/backends/arm/math/sve/funcs_sve.h"
#endif

typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;
typedef paddle::lite::operators::ActivationParam ActivationParam;

DEFINE_int32(power_mode,
             0,
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

DEFINE_int32(relu_type,
             0,
             "relu type, 0: no relu; 1: relu; 2: relu6; 3: leaky_relu;");
DEFINE_bool(flag_bias, true, "with bias");

bool test_gemm_int8(bool tra,
                    bool trb,
                    int m,
                    int n,
                    int k,
                    bool has_bias,
                    int relu_type,
                    int cls,
                    int ths) {
  Tensor ta;
  Tensor tb;
  Tensor tc_int8;
  Tensor tc_sve_int8;
  Tensor tc_fp32;
  Tensor tc_sve_fp32;
  Tensor tc_basic_int8;
  Tensor tc_basic_fp32;
  Tensor tbias;

  ta.Resize({m, k});
  tb.Resize({k, n});
  tc_int8.Resize({m, n});
  tc_sve_int8.Resize({m, n});
  tc_fp32.Resize({m, n});
  tc_sve_fp32.Resize({m, n});
  tc_basic_int8.Resize({m, n});
  tc_basic_fp32.Resize({m, n});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kInt8));
  tb.set_precision(PRECISION(kInt8));
  tc_int8.set_precision(PRECISION(kInt8));
  tc_sve_int8.set_precision(PRECISION(kInt8));
  tc_fp32.set_precision(PRECISION(kFloat));
  tc_sve_fp32.set_precision(PRECISION(kFloat));
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
  ActivationParam act_param;
  bool has_relu = false;
  switch (relu_type) {
    case 0:
      has_relu = false;
      act_param.has_active = has_relu;
      break;
    case 1:
      has_relu = true;

    case 2:
    case 3:
      act_param.has_active = has_relu;
      act_param.active_type = (paddle::lite_api::ActivationType)relu_type;
      break;
    default:
      has_relu = true;
      act_param.has_active = has_relu;
      act_param.active_type = (paddle::lite_api::ActivationType)1;
  }

  for (int j = 0; j < m; ++j) {
    scale_merge_fp32[j] = scale_a[j] * scale_b[0];
    scale_merge_int8[j] = scale_merge_fp32[j] / scale_c[0];
  }

  LOG(INFO) << "gemm_int8 M: " << m << ", N: " << n << ", K: " << k
            << ", transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", relu_type: " << relu_type
            << ", bias: " << (has_bias ? "true" : "false");
#ifdef LITE_WITH_ARM
  int lda = tra ? m : k;
  int ldb = trb ? k : n;
  int ldc = n;

  auto da = ta.mutable_data<int8_t>();
  auto db = tb.mutable_data<int8_t>();
  auto dc_int8 = tc_int8.mutable_data<int8_t>();
  auto dc_sve_int8 = tc_sve_int8.mutable_data<int8_t>();
  auto dc_fp32 = tc_fp32.mutable_data<float>();
  auto dc_sve_fp32 = tc_sve_fp32.mutable_data<float>();
  auto dc_basic_int8 = tc_basic_int8.mutable_data<int8_t>();
  auto dc_basic_fp32 = tc_basic_fp32.mutable_data<float>();
  // set intial input to be 0
  memset(reinterpret_cast<char*>(dc_basic_fp32),
         0,
         tc_basic_fp32.numel() * sizeof(float));
  auto dbias = tbias.mutable_data<float>();

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
  //! prepack
  Tensor tpackedA;
  int hblock = paddle::lite::arm::math::get_hblock_int8(&ctx);
  int round_up_a = ((hblock + m - 1) / hblock) * hblock;
  int round_up_k = 4 * ((k + 3) / 4);
  tpackedA.Resize({round_up_a * round_up_k});
  auto prepack_data = tpackedA.data<int8_t>();

  paddle::lite::arm::math::prepackA_int8(
      tpackedA.mutable_data<int8_t>(), da, lda, 0, m, 0, k, tra, &ctx);
  prepack_data = tpackedA.data<int8_t>();

  /// warmup
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::gemm_prepack_int8(tpackedA.data<int8_t>(),
                                               db,
                                               dbias,
                                               dc_fp32,
                                               m,
                                               n,
                                               k,
                                               has_bias,
                                               trb,
                                               scale_merge_fp32.data(),
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

#ifdef LITE_WITH_ARM8_SVE2
  //! prepack
  Tensor tpackedA_sve;
  int hblock_sve = paddle::lite::arm::math::sve::get_hblock_int8_sve(&ctx);
  int round_up_a_sve = ((hblock_sve + m - 1) / hblock_sve) * hblock_sve;
  int round_up_k_sve = 8 * ((k + 7) / 8);
  tpackedA_sve.Resize({round_up_a_sve * round_up_k_sve});

  paddle::lite::arm::math::sve::prepackA_int8_sve(
      tpackedA_sve.mutable_data<int8_t>(), da, lda, 0, m, 0, k, tra, &ctx);

  // sve
  Timer t1;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    t1.Start();
    // dc_sve_fp32
    paddle::lite::arm::math::sve::gemm_prepack_int8_sve<float>(
        tpackedA_sve.data<int8_t>(),
        db,
        dbias,
        dc_sve_fp32,
        m,
        n,
        k,
        has_bias,
        trb,
        scale_merge_fp32.data(),
        act_param,
        &ctx);
    t1.Stop();
  }

  LOG(INFO) << "sve int8_fp32 M: " << m << ", N: " << n << ", K: " << k
            << ", power_mode: " << cls << ", threads: " << ths
            << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t1.LapTimes().Avg()
            << " ms, min time: " << t1.LapTimes().Min()
            << " ms, mean GOPs: " << ops * 1e-6f / t1.LapTimes().Avg()
            << " GOPs, max GOPs: " << ops * 1e-6f / t1.LapTimes().Min()
            << " GOPs";
  t1.Reset();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    t1.Start();
    paddle::lite::arm::math::sve::gemm_prepack_int8_sve<int8_t>(
        tpackedA_sve.data<int8_t>(),
        db,
        dbias_int8,
        dc_sve_int8,
        m,
        n,
        k,
        has_bias,
        trb,
        scale_merge_int8.data(),
        act_param,
        &ctx);
    t1.Stop();
  }
  LOG(INFO) << "sve int8_int8 M: " << m << ", N: " << n << ", K: " << k
            << ", power_mode: " << cls << ", threads: " << ths
            << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t1.LapTimes().Avg()
            << " ms, min time: " << t1.LapTimes().Min()
            << " ms, mean GOPs: " << ops * 1e-6f / t1.LapTimes().Avg()
            << " GOPs, max GOPs: " << ops * 1e-6f / t1.LapTimes().Min()
            << " GOPs";
#endif

  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    paddle::lite::arm::math::gemm_prepack_int8(tpackedA.data<int8_t>(),
                                               db,
                                               dbias_int8,
                                               dc_int8,
                                               m,
                                               n,
                                               k,
                                               has_bias,
                                               trb,
                                               scale_merge_int8.data(),
                                               act_param,
                                               &ctx);
    t0.Stop();
  }
  LOG(INFO) << "gemm_int8_int8 output: M: " << m << ", N: " << n << ", K: " << k
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
    paddle::lite::arm::math::gemm_prepack_int8(tpackedA.data<int8_t>(),
                                               db,
                                               dbias,
                                               dc_fp32,
                                               m,
                                               n,
                                               k,
                                               has_bias,
                                               trb,
                                               scale_merge_fp32.data(),
                                               act_param,
                                               &ctx);
    t0.Stop();
  }
  LOG(INFO) << "gemm_int8_fp32 output: M: " << m << ", N: " << n << ", K: " << k
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
#ifdef LITE_WITH_ARM8_SVE2
    // fp32
    tensor_cmp_host(tc_basic_fp32, tc_sve_fp32, max_ratio, max_diff);
    LOG(INFO) << "sve fp32 compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio;
    if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
      Tensor tdiff;
      tdiff.set_precision(PRECISION(kFloat));
      tdiff.Resize(tc_sve_fp32.dims());
      tensor_diff(tc_basic_fp32, tc_sve_fp32, tdiff);
      LOG(INFO) << "basic result: ";
      print_tensor(tc_basic_fp32);
      LOG(INFO) << "lite result: ";
      print_tensor(tc_sve_fp32);
      LOG(INFO) << "diff result: ";
      print_tensor(tdiff);
      return false;
    }
    /// int8 result
    max_ratio = 0;
    max_diff = 0;
    tensor_cmp_host(tc_basic_int8, tc_sve_int8, max_ratio, max_diff);
    LOG(INFO) << "sve int8 compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio;
    if (fabs(max_ratio) > 1e-4f) {
      Tensor tdiff;
      tdiff.Resize(tc_sve_int8.dims());
      tdiff.set_precision(PRECISION(kInt8));
      tensor_diff(tc_basic_int8, tc_sve_int8, tdiff);
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
        print_tensor(tc_sve_int8);
        LOG(WARNING) << "int8 diff tensor";
        print_tensor(tdiff);
        return false;
      }
    }
#endif
  }
#endif
  return true;
}

TEST(TestLiteGemmInt8, gemm_prepacked_int8) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    for (auto& m : {1, 3, 6, 8, 32, 33, 34, 35, 38, 41, 397}) {
      for (auto& n : {1, 3, 6, 13, 35, 141, 512, 789}) {
        for (auto& k : {1, 3, 8, 59, 60, 61, 62, 66, 67, 71}) {
          for (auto& tra : {false, true}) {
            for (auto& trb : {true, false}) {
              for (auto& has_bias : {false, true}) {
                for (auto& relu_type : {0, 1}) {
                  for (auto& th : {1, 2, 4}) {
                    auto flag = test_gemm_int8(tra,
                                               trb,
                                               m,
                                               n,
                                               k,
                                               has_bias,
                                               relu_type,
                                               FLAGS_power_mode,
                                               th);
                    if (flag) {
                      LOG(INFO) << "test m = " << m << ", n=" << n
                                << ", k=" << k
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

TEST(TestGemmInt8Custom, gemm_prepacked_int8_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_gemm_int8(FLAGS_traA,
                             FLAGS_traB,
                             FLAGS_M,
                             FLAGS_N,
                             FLAGS_K,
                             FLAGS_flag_bias,
                             FLAGS_relu_type,
                             FLAGS_power_mode,
                             FLAGS_threads);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", k=" << FLAGS_K << ", trans A: " << FLAGS_traA
               << ", trans B: " << FLAGS_traB << ", bias: " << FLAGS_flag_bias
               << ", relu: " << FLAGS_relu_type << " failed!!";
  }
  LOG(INFO) << "test m = " << FLAGS_M << ", n=" << FLAGS_N << ", k=" << FLAGS_K
            << ", trans A: " << FLAGS_traA << ", trans B: " << FLAGS_traB
            << ", bias: " << FLAGS_flag_bias << ", relu: " << FLAGS_relu_type
            << " passed!!";
}
