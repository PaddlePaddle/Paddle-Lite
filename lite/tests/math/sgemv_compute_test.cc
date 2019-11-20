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
#include "lite/core/tensor.h"
#include "lite/tests/utils/tensor_utils.h"
#include "lite/tests/utils/timer.h"

typedef paddle::lite::Tensor Tensor;

DEFINE_int32(cluster, 3, "cluster id");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_bool(basic_test, true, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(M, 512, "sgemv: M");
DEFINE_int32(K, 512, "sgemv: K");

DEFINE_bool(traA, false, "gemv: A transpose");

DEFINE_bool(flag_relu, false, "do relu");
DEFINE_bool(flag_bias, false, "with bias");

bool test_sgemv(
    bool tra, int m, int k, bool has_bias, bool has_relu, int cls, int ths) {
  Tensor ta;
  Tensor tb;
  Tensor tc;
  Tensor tc_basic;
  Tensor tbias;

  ta.Resize({m, k});
  tb.Resize({k, 1});
  tc.Resize({m, 1});
  tc_basic.Resize({m, 1});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kFloat));
  tb.set_precision(PRECISION(kFloat));
  tc.set_precision(PRECISION(kFloat));
  tc_basic.set_precision(PRECISION(kFloat));
  tbias.set_precision(PRECISION(kFloat));

  fill_tensor_rand(ta, -1.f, 1.f);
  // fill_tensor_const(ta, 1.f);
  fill_tensor_rand(tb, -1.f, 1.f);
  // fill_tensor_const(tb, 1.f);
  fill_tensor_rand(tbias, -1.f, 1.f);

  LOG(INFO) << "sgemv M: " << m << ", K: " << k
            << ", transA: " << (tra ? "true" : "false")
            << ", relu: " << (has_relu ? "true" : "false")
            << ", bias: " << (has_bias ? "true" : "false");
#ifdef LITE_WITH_ARM

  auto da = ta.mutable_data<float>();
  auto db = tb.mutable_data<float>();
  auto dc = tc.mutable_data<float>();
  auto dc_basic = tc_basic.mutable_data<float>();
  auto dbias = tbias.mutable_data<float>();

  if (FLAGS_check_result) {
    basic_gemv(
        m, k, da, db, dbias, dc_basic, 1.f, 0.f, tra, has_bias, has_relu);
  }
  paddle::lite::Timer t0;
  //! compute
  double ops = 2.0 * m * k;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);
  /// warmup
  for (int j = 0; j < FLAGS_warmup; ++j) {
    paddle::lite::arm::math::sgemv(
        da, db, dc, tra, m, k, has_bias, dbias, has_relu, &ctx);
  }

  t0.clear();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.start();
    paddle::lite::arm::math::sgemv(
        da, db, dc, tra, m, k, has_bias, dbias, has_relu, &ctx);
    t0.end();
  }
  LOG(INFO) << "gemv output: M: " << m << ", K: " << k << ", cluster: " << cls
            << ", threads: " << ths << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t0.get_average_ms()
            << " ms, min time: " << t0.get_min_time()
            << " ms, mean GOPs: " << ops * 1e-6f / t0.get_average_ms()
            << " GOPs, max GOPs: " << ops * 1e-6f / t0.get_min_time()
            << " GOPs";

  if (FLAGS_check_result) {
    double max_ratio = 0;
    double max_diff = 0;
    /// fp32 result
    tensor_cmp_host(tc_basic, tc, max_ratio, max_diff);
    LOG(INFO) << "compare result, max diff: " << max_diff
              << ", max ratio: " << max_ratio;
    if (std::abs(max_ratio) > 1e-4f && std::abs(max_diff) > 5e-5f) {
      Tensor tdiff;
      tdiff.set_precision(PRECISION(kFloat));
      tdiff.Resize(tc.dims());
      tensor_diff(tc_basic, tc, tdiff);
      LOG(INFO) << "basic result: ";
      print_tensor(tc_basic);
      LOG(INFO) << "saber result: ";
      print_tensor(tc);
      LOG(INFO) << "diff result: ";
      print_tensor(tdiff);
      return false;
    }
  }
#endif
  return true;
}

TEST(TestLiteSgemv, Sgemv) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic sgemv test";
    for (auto& m : {1, 3, 8, 21, 32, 397}) {
      for (auto& k : {1, 3, 8, 17, 59, 234}) {
        for (auto& tra : {true, false}) {
          for (auto& has_bias : {false, true}) {
            for (auto& has_relu : {false, true}) {
              for (auto& th : {1, 2, 4}) {
                auto flag = test_sgemv(
                    tra, m, k, has_bias, has_relu, FLAGS_cluster, th);
                if (flag) {
                  LOG(INFO) << "test m = " << m << ", k=" << k
                            << ", bias: " << (has_bias ? "true" : "false")
                            << ", relu: " << (has_relu ? "true" : "false")
                            << ", trans A: " << (tra ? "true" : "false")
                            << ", threads: " << th << " passed\n";
                } else {
                  LOG(FATAL) << "test m = " << m << ", k=" << k
                             << ", bias: " << (has_bias ? "true" : "false")
                             << ", relu: " << (has_relu ? "true" : "false")
                             << ", trans A: " << (tra ? "true" : "false")
                             << ", threads: " << th << " failed\n";
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(TestSgemvCustom, Sgemv_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_sgemv(FLAGS_traA,
                         FLAGS_M,
                         FLAGS_K,
                         FLAGS_flag_bias,
                         FLAGS_flag_relu,
                         FLAGS_cluster,
                         FLAGS_threads);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", k=" << FLAGS_K
               << ", trans A: " << FLAGS_traA << ", bias: " << FLAGS_flag_bias
               << ", relu: " << FLAGS_flag_relu << " failed!!";
  }
  LOG(INFO) << "test m = " << FLAGS_M << ", k=" << FLAGS_K
            << ", trans A: " << FLAGS_traA << ", bias: " << FLAGS_flag_bias
            << ", relu: " << FLAGS_flag_relu << " passed!!";
}
