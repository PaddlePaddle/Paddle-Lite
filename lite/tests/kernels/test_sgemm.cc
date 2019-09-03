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

//
// Created by Li,Xiaoyang(SYS) on 2019-07-25.
//

#include "lite/tests/kernels/fill_data.h"
#include "lite/tests/kernels/test_funcs.h"
#ifdef LITE_WITH_ARM
#include "lite/backends/arm/math/funcs.h"
#endif
#include "lite/core/context.h"
#include "lite/core/tensor.h"
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

int g_offset_a = 10;
int g_offset_b = 10;
int g_offset_c = 10;

float g_alpha = 1.f;
float g_beta = 0.f;

const int MALLOC_ALIGN = 16;

static void* fast_malloc1(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p;
  p = static_cast<char*>(malloc(offset + size));
  if (!p) {
    return nullptr;
  }
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}

static void fast_free1(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

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
  size_t size_a = tra ? k * lda : m * lda;
  size_t size_b = trb ? n * ldb : k * ldb;

  auto da = static_cast<float*>(fast_malloc1(size_a * sizeof(float)));
  auto db = static_cast<float*>(fast_malloc1(size_b * sizeof(float)));
  auto dc = static_cast<float*>(fast_malloc1(m * ldc * sizeof(float)));
  auto dc_basic = static_cast<float*>(fast_malloc1(m * ldc * sizeof(float)));
  auto dbias = static_cast<float*>(fast_malloc1(m * sizeof(float)));

  fill_data_rand(da, -1.f, 1.f, size_a);
  fill_data_rand(db, -1.f, 1.f, size_b);
  fill_data_rand(dbias, -1.f, 1.f, m);
  fill_data_rand(dc, -1.f, 1.f, m * ldc);
  memcpy(dc_basic, dc, sizeof(float) * m * ldc);

  LOG(INFO) << "sgemm M: " << m << ", N: " << n << ", K: " << k;
  LOG(INFO) << "strides, lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc;
  LOG(INFO) << "alpha: " << alpha << ", beta: " << beta;
  LOG(INFO) << "transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false");
  LOG(INFO) << "relu: " << (has_relu ? "true" : "false")
            << ", bias: " << (has_bias ? "true" : "false");

  LOG(INFO) << "basic sgemm compute";
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

  float max_error = 0.f;
  float max_ratio = 0.f;
#ifdef LITE_WITH_ARM
  //! compute
  LOG(INFO) << "sgemm compute";
  double ops = 2.0 * m * n * k;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();

  paddle::lite::arm::math::sgemm(tra,
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
                                 dc,
                                 ldc,
                                 dbias,
                                 has_bias,
                                 has_relu,
                                 &ctx);

  for (int i = 0; i < m * ldc; ++i) {
    auto error = fabsf(dc[i] - dc_basic[i]);
    if (error > max_error) {
      max_error = error;
      max_ratio = error / fabsf(dc_basic[i]);
    }
  }
  if (max_error > 2e-5f && max_ratio > 2e-5f) {
    LOG(INFO) << "max ratio: " << max_ratio << ", max_error: " << max_error;
    LOG(INFO) << "sgemm result:";
    for (int i = 0; i < m * ldc; ++i) {
      printf("%f ", dc[i]);
      if ((i + 1) % ldc == 0) {
        printf("\n");
      }
    }
    LOG(INFO) << "basic result:";
    for (int i = 0; i < m * ldc; ++i) {
      printf("%f ", dc_basic[i]);
      if ((i + 1) % ldc == 0) {
        printf("\n");
      }
    }
  }
#endif
  fast_free1(da);
  fast_free1(db);
  fast_free1(dbias);
  fast_free1(dc);
  fast_free1(dc_basic);
  return max_error < 2e-5f || max_ratio < 2e-5f;
}

void test_input() {
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

void test_func_sgemm_prepacked() {
  if (g_basic_test) {
    LOG(INFO) << "run basic sgemm test";
    for (auto& m : {1, 8, 16, 111, 256, 397, 512, 777, 1024}) {
      for (auto& n : {1, 3, 13, 141, 256, 345, 512, 789, 1024}) {
        for (auto& k : {1, 4, 15, 59, 128, 234, 512, 678, 1024}) {
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

int main(int argc, const char** argv) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  LOG(ERROR) << "usage: ./" << argv[0]
             << " [do_basic_test] [cluster]  [threads]  [m] [n]  [k] [transA] "
                "[transB] [relu] [bias] [test iter] [compare result]";
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
      LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  "
                                              "[threads]  [m] [n]  [k] "
                                              "[transA] [transB] [bias] [relu] "
                                              "[test iter] [compare result]";
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
  test_input();
  if (g_basic_test) {
    test_func_sgemm_prepacked();
  }
  return 0;
}
