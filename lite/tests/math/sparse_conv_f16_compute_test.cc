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
#include "lite/tests/math/conv_ut.h"
#include "lite/tests/utils/tensor_utils.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ActivationParam ActivationParam;
using paddle::lite::profile::Timer;

DEFINE_int32(M, 512, "spmm: M");
DEFINE_int32(N, 512, "spmm: N");
DEFINE_int32(K, 512, "spmm: K");

DEFINE_bool(traA, false, "spmm: A transpose");
DEFINE_bool(traB, false, "spmm: B transpose");

DEFINE_int32(offset_a, 0, "A offset");
DEFINE_int32(offset_b, 0, "B offset");
DEFINE_int32(offset_c, 0, "C offset");

DEFINE_double(alpha, 1.0, "alpha");
DEFINE_double(beta, 0.0, "beta");
DEFINE_bool(flag_semi, false, "do semi");

DEFINE_double(flag_sparsity, 0.8, "with sparsity");

#ifdef LITE_WITH_ARM
bool test_spmm_fp16(bool tra,
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
                    int flag_act,
                    bool has_semi,
                    int cls,
                    int ths,
                    float sparsity,
                    float six = 6.f,
                    float scale = 1.f) {
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
  tc.Resize({m, ldc});
  tc_basic.Resize({m, ldc});
  tc_backup.Resize({m, ldc});
  tbias.Resize({m});

  ta.set_precision(PRECISION(kFP16));
  tb.set_precision(PRECISION(kFP16));
  tc.set_precision(PRECISION(kFP16));
  tc_basic.set_precision(PRECISION(kFP16));
  tc_backup.set_precision(PRECISION(kFP16));
  tbias.set_precision(PRECISION(kFP16));

  fill_tensor_rand(ta, -1.f, 1.f);
  fill_tensor_rand(tb, -1.f, 1.f);
  fill_tensor_rand(tbias, -1.f, 1.f);
  fill_tensor_rand(tc, -1.f, 1.f);

  auto da = ta.mutable_data<float16_t>();
  auto db = tb.mutable_data<float16_t>();
  auto dc = tc.mutable_data<float16_t>();
  auto dc_basic = tc_basic.mutable_data<float16_t>();
  auto dc_backup = tc_backup.mutable_data<float16_t>();
  auto dbias = tbias.mutable_data<float16_t>();

  if (has_semi) {
    int para_h = m;
    int para_w = k;
    int two_num = para_h / 2 * para_w;
    int one_num = para_h % 2 * para_w;
    int sparse_num = two_num * sparsity;
    for (int i = 0; i < para_h / 2; i++) {
      for (int j = 0; j < para_w; j++) {
        if (((da[i] + 1) / 2.0f) <= sparsity) {
          da[i * 2 * para_w + j] = 0.f;
          da[(i * 2 + 1) * para_w + j] = 0.f;
        }
      }
    }
    if (one_num > 0) {
      for (int i = (para_h / 2 * 2) * para_w; i < para_h * para_w; i++) {
        if (((da[i] + 1) / 2.0f) <= sparsity) {
          da[i] = 0.f;
        }
      }
    }
  } else {
    for (int i = 0; i < size_a; i++) {
      if (((da[i] + 1) / 2.0f) < sparsity) {
        da[i] = 0.0f;
      }
    }
  }
  ActivationParam act_param;
  paddle::lite_api::ActivationType act =
      paddle::lite_api::ActivationType::kIndentity;
  if (flag_act == 1) {
    act = paddle::lite_api::ActivationType::kRelu;
  } else if (flag_act == 2) {
    act = paddle::lite_api::ActivationType::kRelu6;
    act_param.Relu_clipped_coef = six;
  } else if (flag_act == 4) {
    act = paddle::lite_api::ActivationType::kLeakyRelu;
    act_param.Leaky_relu_alpha = scale;
  } else if (flag_act == 10) {
    act = paddle::lite_api::ActivationType::kHardSwish;
    act_param.hard_swish_scale = scale;
    act_param.hard_swish_offset = 1.f;
    act_param.hard_swish_threshold = 6.f;
  }
  act_param.active_type = act;
  act_param.has_active = (flag_act > 0);

  memcpy(dc_basic, dc, sizeof(float16_t) * m * ldc);
  memcpy(dc_backup, dc, sizeof(float16_t) * m * ldc);

  VLOG(4) << "spmm M: " << m << ", N: " << n << ", K: " << k
          << ", strides, lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
          << ", alpha: " << alpha << ", beta: " << beta
          << ", transA: " << (tra ? "true" : "false")
          << ", transB: " << (trb ? "true" : "false")
          << ", flag_act: " << flag_act
          << ", semi: " << (has_semi ? "true" : "false")
          << ", bias: " << (has_bias ? "true" : "false")
          << ",sparsity: " << sparsity << ", threads: " << ths;
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
               flag_act,
               six,
               act_param.Leaky_relu_alpha,
               act_param.hard_swish_scale,
               act_param.hard_swish_offset,
               act_param.hard_swish_threshold);
  }
  Timer t0;
  int count_nonzeroes = 0;
  int count_channels = 0;
  int count_blocks = 0;
  int f_semi = 0;
  int num_build_nonzeroes = 0;
  int zero_num = 0;
  int ch_out = m;
  int ch_in = k;
  int im_size = n;
  int weight_num = m * k;
  zero_num = ComputeSemiSparseZeros<float16_t>(&ta,
                                               &count_nonzeroes,
                                               &count_channels,
                                               &count_blocks,
                                               &f_semi,
                                               ch_out,
                                               ch_in);
  if (f_semi == 0) {
    zero_num =
        ComputeSparseZeros<float16_t>(&ta, &num_build_nonzeroes, ch_out, ch_in);
  }
  int nonzero_num = weight_num - zero_num;
  if (nonzero_num <= 0) {
    return true;
  }
  Tensor nonzeros_output_t;
  Tensor oc_nonzeros_t;
  Tensor ic_diffs_t;
  if (f_semi == 1) {
    nonzeros_output_t.Resize({count_nonzeroes});
    oc_nonzeros_t.Resize({ch_out});
    ic_diffs_t.Resize({count_blocks});
  } else {
    nonzeros_output_t.Resize({num_build_nonzeroes});
    oc_nonzeros_t.Resize({ch_out});
    ic_diffs_t.Resize({num_build_nonzeroes});
  }
  int first_ic = 0;
  if (f_semi == 1) {
    first_ic = ComputeSemiSparseWeight<float16_t>(&ta,
                                                  ch_out,
                                                  ch_in,
                                                  im_size,
                                                  count_nonzeroes,
                                                  count_channels,
                                                  count_blocks,
                                                  &nonzeros_output_t,
                                                  &oc_nonzeros_t,
                                                  &ic_diffs_t);
  } else {
    first_ic = ComputeSparseWeight<float16_t>(&ta,
                                              ch_out,
                                              ch_in,
                                              im_size,
                                              nonzero_num,
                                              num_build_nonzeroes,
                                              &nonzeros_output_t,
                                              &oc_nonzeros_t,
                                              &ic_diffs_t);
  }

  double ops = 2.0 * m * n * k;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);

  const float16_t* input = tb.data<float16_t>();
  const float16_t* nonzero_weights = nonzeros_output_t.data<float16_t>();
  const int32_t* diffs = ic_diffs_t.data<int32_t>();
  const uint32_t* oc_nonzeros = oc_nonzeros_t.data<uint32_t>();
  const float16_t* bias = has_bias ? tbias.data<float16_t>() : nullptr;
  float16_t* dout = tc.mutable_data<float16_t>();
  const float16_t* din = input + first_ic * im_size;
  int ic = k;
  int oc = m;
  paddle::lite::operators::SparseConvParam param;
  param.activation_param = act_param;
  for (int j = 0; j < FLAGS_warmup; ++j) {
    if (f_semi == 1) {
      paddle::lite::arm::math::fp16::sparse_semi_conv_fp16_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias,
          dout,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    } else {
      paddle::lite::arm::math::fp16::sparse_conv_fp16_pipelined(nonzero_weights,
                                                                din,
                                                                diffs,
                                                                oc_nonzeros,
                                                                bias,
                                                                dout,
                                                                oc,
                                                                ic,
                                                                im_size,
                                                                param,
                                                                &ctx);
    }
  }

  for (int i = 0; i < FLAGS_repeats; ++i) {
    if (i == FLAGS_repeats - 1) {
      memcpy(dc, dc_backup, sizeof(float16_t) * m * ldc);
    }
    t0.Start();
    if (f_semi == 1) {
      paddle::lite::arm::math::fp16::sparse_semi_conv_fp16_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias,
          dout,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    } else {
      paddle::lite::arm::math::fp16::sparse_conv_fp16_pipelined(nonzero_weights,
                                                                din,
                                                                diffs,
                                                                oc_nonzeros,
                                                                bias,
                                                                dout,
                                                                oc,
                                                                ic,
                                                                im_size,
                                                                param,
                                                                &ctx);
    }
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
      int64_t size = tc_basic.numel();
      int count = 0;
      bool check = false;
      auto dc_basic_ptr = tc_basic.data<float16_t>();
      auto dc_ptr = tc.data<float16_t>();
      for (int i = 0; i < size; i++) {
        if (fabs(dc_basic_ptr[i] - dc_ptr[i]) > 1e-1f &&
            fabs(dc_basic_ptr[i] - dc_ptr[i]) /
                    (fmax(fabs(dc_basic_ptr[i]), fabs(dc_ptr[i]))) >
                0.5) {
          check = true;
          LOG(INFO) << "i: " << i << ", dc_basic_ptr: " << dc_basic_ptr[i]
                    << ", dc_ptr: " << dc_ptr[i];
          break;
        }
      }
      if (!check) return true;
      Tensor tdiff;
      tdiff.set_precision(PRECISION(kFP16));
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
  return true;
}
#else
bool test_spmm_fp16(bool tra,
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
                    int flag_act,
                    bool has_semi,
                    int cls,
                    int ths,
                    float sparsity) {
  return true;
}
#endif

TEST(TestSpmmF16, test_func_spmm_f16) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic spmm test";
    for (auto& m : {1, 16, 64, 128}) {
      for (auto& n : {1, 32, 128, 256}) {
        for (auto& k : {1, 3, 8, 59, 234}) {
          for (auto& tra : {false}) {
            for (auto& trb : {false}) {
              for (auto& alpha : {1.f}) {
                for (auto& beta : {0.f}) {
                  for (auto& offset : {0}) {
                    for (auto& has_bias : {false, true}) {
                      for (auto& flag_act : {0, 1, 2, 4, 10}) {
                        for (auto& has_semi : {false, true}) {
                          for (auto& th : {4}) {
                            for (auto& sp : {0.5, 0.7, 0.8}) {
                              int lda = k + offset;
                              if (tra) {
                                lda = m + offset;
                              }
                              int ldb = n + offset;
                              if (trb) {
                                ldb = k + offset;
                              }
                              int ldc = n + offset;
                              auto flag = test_spmm_fp16(tra,
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
                                                         flag_act,
                                                         has_semi,
                                                         FLAGS_power_mode,
                                                         th,
                                                         sp);
                              if (flag) {
                                VLOG(4)
                                    << "test m = " << m << ", n=" << n
                                    << ", k=" << k << ", bias: "
                                    << (has_bias ? "true" : "false")
                                    << ", act: " << flag_act << ", semi: "
                                    << (has_semi ? "true" : "false")
                                    << ", trans A: " << (tra ? "true" : "false")
                                    << ", trans B: " << (trb ? "true" : "false")
                                    << " passed\n";
                              } else {
                                LOG(FATAL)
                                    << "test m = " << m << ", n=" << n
                                    << ", k=" << k << ", bias: "
                                    << (has_bias ? "true" : "false")
                                    << ", act: " << flag_act << ", semi: "
                                    << (has_semi ? "true" : "false")
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
  }
}

TEST(TestSpmmF16Custom, test_func_spmm_f16_custom) {
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
  LOG(INFO) << "run basic spmm test";
  auto flag = test_spmm_fp16(FLAGS_traA,
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
                             FLAGS_flag_act,
                             FLAGS_flag_semi,
                             FLAGS_power_mode,
                             FLAGS_threads,
                             FLAGS_flag_sparsity);
  if (!flag) {
    LOG(FATAL) << "test m = " << FLAGS_M << ", n=" << FLAGS_N
               << ", k=" << FLAGS_K << ", trans A: " << FLAGS_traA
               << ", trans B: " << FLAGS_traB << ", bias: " << FLAGS_flag_bias
               << ", flag_act: " << FLAGS_flag_act << " failed!!";
  }
  LOG(INFO) << "test m = " << FLAGS_M << ", n=" << FLAGS_N << ", k=" << FLAGS_K
            << ", trans A: " << FLAGS_traA << ", trans B: " << FLAGS_traB
            << ", bias: " << FLAGS_flag_bias << ", flag_act: " << FLAGS_flag_act
            << " passed!!";
}
