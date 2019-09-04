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

#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/test_funcs.h"
#ifdef LITE_WITH_ARM
#include "lite/backends/arm/math/funcs.h"
#endif
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/tests/utils/tensor_utils.h"
#include "lite/tests/utils/test_lite.h"

typedef paddle::lite::Tensor Tensor;
typedef lite::test::TestLite TestLite;

int g_cluster = 0;
int g_threads = 1;

bool g_basic_test = false;

int g_M = 512;
int g_N = 512;
int g_K = 512;
bool g_traA = false;
bool g_traB = false;
bool g_flag_relu = false;
bool g_flag_bias = false;
int g_test_iter = 1;
int g_warmup_iter = 0;
bool g_compare_result = true;

int g_offset_a = 0;
int g_offset_b = 0;
int g_offset_c = 0;

float g_alpha = 1.f;
float g_beta = 0.f;

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

  LOG(INFO) << "sgemm M: " << m << ", N: " << n << ", K: " << k
            << ", strides, lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
            << ", alpha: " << alpha << ", beta: " << beta
            << ", transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", relu: " << (has_relu ? "true" : "false")
            << ", bias: " << (has_bias ? "true" : "false");
  if (g_compare_result) {
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
  lite::test::Timer t0;
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
  for (int j = 0; j < g_warmup_iter; ++j) {
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
                                           has_relu,
                                           &ctx);
  }

  for (int i = 0; i < g_test_iter; ++i) {
    if (i == g_test_iter - 1) {
      memcpy(dc, dc_backup, sizeof(float) * m * ldc);
    }
    t0.start();
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
                                           has_relu,
                                           &ctx);
    t0.end();
  }
  LOG(INFO) << "M: " << m << ", N: " << n << ", K: " << k
            << ", cluster: " << cls << ", threads: " << ths
            << ", GOPS: " << ops * 1e-9f
            << " GOPS, avg time: " << t0.get_average_ms()
            << " ms, min time: " << t0.get_min_time()
            << " ms, mean GOPs: " << ops * 1e-6f / t0.get_average_ms()
            << " GOPs, max GOPs: " << ops * 1e-6f / t0.get_min_time()
            << " GOPs";

  if (g_compare_result) {
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

TEST(TestLite, test_func_sgemm_prepacked) {
  if (g_basic_test) {
    LOG(INFO) << "run basic sgemm test";
    for (auto& m : {1, 3, 8, 32, 397}) {
      for (auto& n : {1, 3, 13, 141, 512, 789, 1234, 6789}) {
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
                                                 g_cluster,
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

TEST(TestLite, test_func_sgemm_prepacked_custom) {
  int lda = g_K + g_offset_a;
  if (g_traA) {
    lda = g_M + g_offset_a;
  }
  int ldb = g_N + g_offset_b;
  if (g_traB) {
    ldb = g_K + g_offset_b;
  }
  int ldc = g_N + g_offset_c;
  auto flag = test_sgemm(g_traA,
                         g_traB,
                         g_M,
                         g_N,
                         g_K,
                         lda,
                         ldb,
                         ldc,
                         g_alpha,
                         g_beta,
                         g_flag_bias,
                         g_flag_relu,
                         g_cluster,
                         g_threads);
  if (!flag) {
    LOG(FATAL) << "test m = " << g_M << ", n=" << g_N << ", k=" << g_K
               << ", trans A: " << g_traA << ", trans B: " << g_traB
               << ", bias: " << g_flag_bias << ", relu: " << g_flag_relu
               << " failed!!";
  }
  LOG(INFO) << "test m = " << g_M << ", n=" << g_N << ", k=" << g_K
            << ", trans A: " << g_traA << ", trans B: " << g_traB
            << ", bias: " << g_flag_bias << ", relu: " << g_flag_relu
            << " passed!!";
}

int main(int argc, const char** argv) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  LOG(ERROR)
      << "usage: ./" << argv[0]
      << " [do_basic_test] [cluster]  [threads]  [m] [n]  [k] [transA] "
         "[transB] [relu] [bias] [test iter] [compare result] [warm_up iter]"
         "[a_offset] [b_offset] [c_offset] [alpha] [beta]";
  if (argc > 1) {
    g_basic_test = atoi(argv[1]) > 0;
  }
  if (argc > 2) {
    g_cluster = atoi(argv[2]);
  }
  if (argc > 3) {
    g_threads = atoi(argv[3]);
  }
  if (argc > 4) {
    if (argc < 10) {
      LOG(ERROR) << "usage: ./" << argv[0]
                 << " [do_basic_test] [cluster]  "
                    "[threads]  [m] [n]  [k] "
                    "[transA] [transB] [bias] [relu] "
                    "[test iter] [compare result] [warm_up iter]"
                    "[a_offset] [b_offset] [c_offset] [alpha] [beta]";
      return 0;
    }
    g_M = atoi(argv[4]);
    g_N = atoi(argv[5]);
    g_K = atoi(argv[6]);
    g_traA = atoi(argv[7]) > 0;
    g_traB = atoi(argv[8]) > 0;
    g_flag_bias = atoi(argv[9]) > 0;
    g_flag_relu = atoi(argv[10]) > 0;
  }
  if (argc > 11) {
    g_test_iter = atoi(argv[11]);
  }
  if (argc > 12) {
    g_compare_result = atoi(argv[12]) > 0;
  }
  if (argc > 13) {
    g_warmup_iter = atoi(argv[13]);
  }
  if (argc > 14) {
    g_offset_a = atoi(argv[14]);
  }
  if (argc > 15) {
    g_offset_b = atoi(argv[15]);
  }
  if (argc > 16) {
    g_offset_c = atoi(argv[16]);
  }
  if (argc > 17) {
    g_alpha = atof(argv[17]);
  }
  if (argc > 18) {
    g_beta = atof(argv[18]);
  }
  InitTest();
  RUN_ALL_TESTS(argv[0]);
  return 0;
}
