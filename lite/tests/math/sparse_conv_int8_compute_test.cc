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

typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;
typedef paddle::lite::operators::ActivationParam ActivationParam;

DEFINE_int32(M, 512, "spmm: M");
DEFINE_int32(N, 512, "spmm: N");
DEFINE_int32(K, 512, "spmm: K");

DEFINE_bool(traA, false, "spmm: A transpose");
DEFINE_bool(traB, false, "spmm: B transpose");

DEFINE_int32(relu_type,
             0,
             "relu type, 0: no relu; 1: relu; 2: relu6; 3: leaky_relu;");
DEFINE_bool(flag_semi, false, "with semi");

DEFINE_double(flag_sparsity, 0.8, "with sparsity");

#ifdef LITE_WITH_ARM
bool test_spmm_int8(bool tra,
                    bool trb,
                    int m,
                    int n,
                    int k,
                    bool has_bias,
                    bool has_semi,
                    int relu_type,
                    int cls,
                    int ths,
                    float sparsity) {
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

  auto da8 = ta.mutable_data<int8_t>();
  if (has_semi) {
    int para_h = m;
    int para_w = k;
    int two_num = para_h / 2 * para_w;
    int one_num = para_h % 2 * para_w;
    int sparse_num = two_num * sparsity;
    for (int i = 0; i < para_h / 2; i++) {
      for (int j = 0; j < para_w; j++) {
        if (((da8[i] + 128) / 255.0f) <= sparsity) {
          da8[i * 2 * para_w + j] = 0;
          da8[(i * 2 + 1) * para_w + j] = 0;
        }
      }
    }
    if (one_num > 0) {
      for (int i = (para_h / 2 * 2) * para_w; i < para_h * para_w; i++) {
        if (((da8[i] + 128) / 255.0f) <= sparsity) {
          da8[i] = 0;
        }
      }
    }
  } else {
    for (int i = 0; i < m * k; i++) {
      if (((da8[i] + 128) / 255.0f) < sparsity) {
        da8[i] = 0;
      }
    }
  }

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

  LOG(INFO) << "spmm_int8 M: " << m << ", N: " << n << ", K: " << k
            << ", transA: " << (tra ? "true" : "false")
            << ", transB: " << (trb ? "true" : "false")
            << ", relu_type: " << relu_type
            << ", bias: " << (has_bias ? "true" : "false")
            << ", semi: " << (has_semi ? "true" : "false");

  int lda = tra ? m : k;
  int ldb = trb ? k : n;
  int ldc = n;

  auto da = ta.mutable_data<int8_t>();
  auto db = tb.mutable_data<int8_t>();
  auto dc_int8 = tc_int8.mutable_data<int8_t>();
  auto dc_fp32 = tc_fp32.mutable_data<float>();
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
  int zero_num = 0;
  int num_build_nonzeroes = 0;
  int count_nonzeroes = 0;
  int count_channels = 0;
  int count_blocks = 0;
  int f_semi = 0;
  int ch_out = m;
  int ch_in = k;
  int im_size = n;
  int weight_num = m * k;
  zero_num = ComputeSemiSparseZeros<int8_t>(&ta,
                                            &count_nonzeroes,
                                            &count_channels,
                                            &count_blocks,
                                            &f_semi,
                                            ch_out,
                                            ch_in);
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
    nonzeros_output_t.Resize({count_nonzeroes});
    oc_nonzeros_t.Resize({ch_out});
    ic_diffs_t.Resize({count_nonzeroes});
  }
  int first_ic = 0;
  if (f_semi == 1) {
    first_ic = ComputeSemiSparseWeight<int8_t>(&ta,
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
    first_ic = ComputeSparseWeight<int8_t>(&ta,
                                           ch_out,
                                           ch_in,
                                           im_size,
                                           nonzero_num,
                                           &nonzeros_output_t,
                                           &oc_nonzeros_t,
                                           &ic_diffs_t);
  }
  Timer t0;
  //! compute
  double ops = 2.0 * m * n * k;
  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ths = ((f_semi == 1) ? 1 : ths);
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), ths);

  const int8_t* input = tb.data<int8_t>();
  const int8_t* nonzero_weights = nonzeros_output_t.data<int8_t>();
  const int32_t* diffs = ic_diffs_t.data<int32_t>();
  const uint32_t* oc_nonzeros = oc_nonzeros_t.data<uint32_t>();
  const float* bias_f32 = has_bias ? tbias.data<float>() : nullptr;
  float* dout_f32 = tc_fp32.mutable_data<float>();
  const int8_t* din = input + first_ic * im_size;
  int ic = k;
  int oc = m;
  paddle::lite::operators::SparseConvParam param;
  param.activation_param = act_param;
  /// warmup
  for (int j = 0; j < FLAGS_warmup; ++j) {
    if (f_semi == 1) {
      paddle::lite::arm::math::sparse_semi_conv_int8_fp32_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias_f32,
          scale_merge_fp32.data(),
          dout_f32,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    } else {
      paddle::lite::arm::math::sparse_conv_int8_fp32_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias_f32,
          scale_merge_fp32.data(),
          dout_f32,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    }
  }

  /// int8 output compute
  Tensor tbias_int8;
  tbias_int8.Resize(tbias.dims());
  tbias_int8.set_precision(PRECISION(kFloat));
  auto dbias_int8 = tbias_int8.mutable_data<float>();
  for (int l = 0; l < tbias_int8.numel(); ++l) {
    dbias_int8[l] = dbias[l] / scale_c[0];
  }
  const float* bias_int8 = has_bias ? tbias_int8.data<float>() : nullptr;
  int8_t* dout_int8 = tc_int8.mutable_data<int8_t>();

  for (int i = 0; i < FLAGS_repeats; ++i) {
    t0.Start();
    if (f_semi == 1) {
      paddle::lite::arm::math::sparse_semi_conv_int8_int8_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias_int8,
          scale_merge_int8.data(),
          dout_int8,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    } else {
      paddle::lite::arm::math::sparse_conv_int8_int8_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias_int8,
          scale_merge_int8.data(),
          dout_int8,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    }
    t0.Stop();
  }
  LOG(INFO) << "spmm_int8_int8 output: M: " << m << ", N: " << n << ", K: " << k
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
    if (f_semi == 1) {
      paddle::lite::arm::math::sparse_semi_conv_int8_fp32_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias_f32,
          scale_merge_fp32.data(),
          dout_f32,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    } else {
      paddle::lite::arm::math::sparse_conv_int8_fp32_pipelined(
          nonzero_weights,
          din,
          diffs,
          oc_nonzeros,
          bias_f32,
          scale_merge_fp32.data(),
          dout_f32,
          oc,
          ic,
          im_size,
          param,
          &ctx);
    }
    t0.Stop();
  }
  LOG(INFO) << "spmm_int8_fp32 output: M: " << m << ", N: " << n << ", K: " << k
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
  }
  return true;
}
#else
bool test_spmm_int8(bool tra,
                    bool trb,
                    int m,
                    int n,
                    int k,
                    bool has_bias,
                    bool has_semi,
                    int relu_type,
                    int cls,
                    int ths,
                    float sparsity) {
  return true;
}
#endif

TEST(TestLiteSpmmInt8, spmm_prepacked_int8) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    for (auto& m : {1, 16, 64, 128}) {
      for (auto& n : {1, 32, 128, 256}) {
        for (auto& k : {1, 109, 512}) {
          for (auto& tra : {false}) {
            for (auto& trb : {false}) {
              for (auto& has_bias : {false, true}) {
                for (auto& has_semi : {false, true}) {
                  for (auto& relu_type : {0, 1}) {
                    for (auto& th : {1, 2, 4}) {
                      for (auto& sp : {0.5, 0.7, 0.8}) {
                        auto flag = test_spmm_int8(tra,
                                                   trb,
                                                   m,
                                                   n,
                                                   k,
                                                   has_bias,
                                                   has_semi,
                                                   relu_type,
                                                   FLAGS_power_mode,
                                                   th,
                                                   sp);
                        if (flag) {
                          VLOG(4) << "test m = " << m << ", n=" << n
                                  << ", k=" << k
                                  << ", bias: " << (has_bias ? "true" : "false")
                                  << ", semi: " << (has_semi ? "true" : "false")
                                  << ", relu: " << relu_type
                                  << ", trans A: " << (tra ? "true" : "false")
                                  << ", trans B: " << (trb ? "true" : "false")
                                  << " passed\n";
                        } else {
                          LOG(FATAL)
                              << "test m = " << m << ", n=" << n << ", k=" << k
                              << ", bias: " << (has_bias ? "true" : "false")
                              << ", semi: " << (has_semi ? "true" : "false")
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
  }
}

TEST(TestSpmmInt8Custom, spmm_prepacked_int8_custom) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  auto flag = test_spmm_int8(FLAGS_traA,
                             FLAGS_traB,
                             FLAGS_M,
                             FLAGS_N,
                             FLAGS_K,
                             FLAGS_flag_bias,
                             FLAGS_flag_semi,
                             FLAGS_relu_type,
                             FLAGS_power_mode,
                             FLAGS_threads,
                             FLAGS_flag_sparsity);
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
