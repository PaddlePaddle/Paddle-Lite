// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef LITE_WITH_X86

#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include <algorithm>
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/gemm_s8u8_compute.h"
#include <string.h>

typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;
typedef paddle::lite::operators::ActivationParam ActivationParam;

void convert_fp32_to_int8(int m, int n, float *input, float scale, int8_t *output, 
    bool is_relu = false, const float *bias = nullptr) {
  int tmp = 0;
  float tmpf = 0.f;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      int offt = i * n + j;
      tmpf = input[offt];
      if(bias != nullptr)
        tmpf += bias[i];
      tmpf = tmpf * scale;
      tmpf = (tmpf >= 0.f) ? (tmpf + 0.5f) : (tmpf - 0.5f);
      tmp = static_cast<int>(tmpf);
      if(is_relu)
        tmp = std::min(std::max(tmp, 0), 127);
      else
        tmp = std::min(std::max(tmp, -127), 127);
      output[offt] = static_cast<int8_t>(tmp);
    }
  }
}

void basic_gemm_fp32(bool traA, bool traB, int M, int N, int K, 
    float *A, int lda, float *B, int ldb, float *C, int ldc) {
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::X86Context>();
  paddle::lite::x86::math::Blas<paddle::lite::TargetType::kX86> matmul(ctx);
  matmul.GEMM<float>(traA,
                     traB,
                     M,
                     N,
                     K,
                     1.f,
                     A,
                     lda,
                     B,
                     ldb,
                     0.f,
                     C,
                     ldc);
}

bool test_gemm_s8u8(bool tra,
                    bool trb,
                    int m,
                    int n,
                    int k,
                    bool has_bias,
                    bool has_relu) {
  Tensor ta, tb, tc, ta_f32, tb_f32, tc_basic_f32, tc_basic_s8;
  Tensor tbias, Sa;

  ta.Resize({m, k});
  tb.Resize({n, k});
  tc.Resize({m, n});
  ta_f32.Resize({m, k});
  tb_f32.Resize({n, k});
  tc_basic_f32.Resize({m, n});
  tc_basic_s8.Resize({m, n});
  tbias.Resize({m});
  Sa.Resize({m});
  ta.set_precision(PRECISION(kInt8));
  tb.set_precision(PRECISION(kInt8));
  tc.set_precision(PRECISION(kInt8));
  ta_f32.set_precision(PRECISION(kFloat));
  tb_f32.set_precision(PRECISION(kFloat));
  tc_basic_f32.set_precision(PRECISION(kFloat));
  tc_basic_s8.set_precision(PRECISION(kInt8));
  tbias.set_precision(PRECISION(kFloat));
  Sa.set_precision(PRECISION(kFloat));

  // input
  fill_tensor_rand(ta, -63, 63);
  fill_tensor_rand(tb, -127, 127);
  if(has_bias)
    fill_tensor_rand(tbias, -1.f, 1.f);
  else
    fill_tensor_rand(tbias, 0, 0);

  // Scale
  auto sa_ptr = Sa.mutable_data<float>();
  for(int i = 0; i < m; i++)
    sa_ptr[i] = 32 * (rand() % 2 + 1);
  float Sb = 127.f;
  float Sc = 127.f;

  int lda = tra ? m : k;
  int ldb = trb ? k : n;
  int ldc = n;

  auto a_ptr_f32 = ta_f32.mutable_data<float>();
  for(int i = 0; i < m; i++) {
    float ssa = sa_ptr[i];
    for(int j = 0; j < k; j++) {
      int offt = tra ? (j * m + i) : (i * k + j);
      a_ptr_f32[offt] = (ta.data<int8_t>())[offt] * 1.f / ssa;
    }
  }
  auto b_ptr_f32 = tb_f32.mutable_data<float>();
  for(int i = 0; i < n * k; i++)
    b_ptr_f32[i] = (tb.data<int8_t>())[i] * 1.f / Sb;
  
  auto c_ptr_test = tc.mutable_data<int8_t>();
  auto c_ptr_basic = tc_basic_s8.mutable_data<int8_t>();
  auto c_ptr_basic_f = tc_basic_f32.mutable_data<float>();

  LOG(INFO) << "gemm_int8 M: " << m << ", N: " << n << ", K: " << k
            << ", transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", has_relu: " << (has_relu ? "true" : "false")
            << ", bias: " << (has_bias ? "true" : "false");

  auto bbias = tbias.data<float>();
  auto tad = ta.data<int8_t>();
  auto tbd = tb.data<int8_t>();

  Timer t0, t1;
  int repeat = 10;
  int warm_up = 10;

  // warm_up
  for(int i = 0; i < warm_up; i++) {
    basic_gemm_fp32(tra, trb, m, n, k,
      a_ptr_f32, lda, b_ptr_f32, ldb, c_ptr_basic_f, ldc);
    memset(c_ptr_basic_f, 0, m * n * 4);
  }

  // base_line
  for(int i = 0; i < repeat; i++) {
    memset(c_ptr_basic_f, 0, m * n * 4);
    t0.Start();
    basic_gemm_fp32(tra, trb, m, n, k,
      a_ptr_f32, lda, b_ptr_f32, ldb, c_ptr_basic_f, ldc);
    t0.Stop();
  }
  convert_fp32_to_int8(m, n, c_ptr_basic_f, Sc, c_ptr_basic, has_relu, bbias);
  
  // test
  int relu_type = has_relu ? 1 : 0;
  paddle::lite::x86::math::generate_gemm_s8u8_x86_kern 
      gemm(tra, trb, m, n, k, tad, tbd, c_ptr_test, n, sa_ptr, Sb, Sc, bbias, relu_type, 1.f);
  for(int i = 0; i < repeat; i++) {
    t1.Start();
    gemm.compute();
    t1.Stop();
  }

  int max_err = 0;
  int at_m = 0;
  int at_n = 0;
  int ll = 0;
  int lr = 0;
  for(int i = 0; i < m * n; i++) {
    if(std::abs(c_ptr_test[i] - c_ptr_basic[i]) > max_err) {
      max_err = std::abs(c_ptr_test[i] - c_ptr_basic[i]);
      at_m = (i / n) + 1;
      at_n = i % m;
      ll = c_ptr_basic[i];
      lr = c_ptr_test[i];
    }
  }
  LOG(INFO) << "precision_diff at "<< at_m << ", "<< at_n << ", " << 
      "real is " << ll << ", test is " << lr;
  LOG(INFO) << "Float avg time(ms): " << t0.LapTimes().Avg() << "  min time(ms)" << t0.LapTimes().Min();
  LOG(INFO) << "S8u8 avg time(ms): " << t1.LapTimes().Avg() << "  min time(ms)" << t1.LapTimes().Min();

  if(max_err > 1)
    return false;
  return true;
}

TEST(TestX86LiteGemmInt8, gemm_s8u8_compute) {
  pthread_t tid = { 0 };
  cpu_set_t cpu_info = { 0 };	
  tid = pthread_self();
  CPU_ZERO(&cpu_info);
  CPU_SET(0, &cpu_info);
  if (0 != pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpu_info)) {
      LOG(FATAL) << "set affinity failed";
  }

  auto flag = test_gemm_s8u8(false,
                             false,
                             512,
                             256,
                             512,
                             false,
                             false);
  if(!flag)
    LOG(FATAL) << "precision check failed (diff > 1)!";
}

#endif // LITE_WITH_X86
