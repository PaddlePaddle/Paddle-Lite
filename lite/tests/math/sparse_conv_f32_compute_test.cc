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
// spmm_test wiil not be operated except that it's
// on arm backend.
DEFINE_bool(basic_test, true, "do all tests");
#else
DEFINE_bool(basic_test, false, "do all tests");
#endif

DEFINE_bool(check_result, true, "check the result");

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

DEFINE_bool(flag_relu, false, "do relu");
DEFINE_bool(flag_bias, false, "with bias");
DEFINE_bool(flag_semi, false, "do semi");

DEFINE_double(flag_sparsity, 0.8, "with sparsity");

template <typename T>
int ComputeSparseZeros(const Tensor* weights,
                       int* num_build_nonzeroes,
                       const int height,
                       const int width) {
  const T* data = weights->data<T>();
  int num_nonzeroes = 0;
  int num_nonzeroes_act = 0;
  for (int i = 0; i < height; i++) {
    int line_nonzeroes = 0;
    for (int j = 0; j < width; j++) {
      if (data[i * width + j] != static_cast<T>(0)) {
        line_nonzeroes++;
      }
    }
    if (line_nonzeroes % 4 == 0) {
      num_nonzeroes += line_nonzeroes;
    } else {
      num_nonzeroes += line_nonzeroes + 4 - (line_nonzeroes % 4);
    }
    num_nonzeroes_act += line_nonzeroes;
  }
  *num_build_nonzeroes = num_nonzeroes;
  return height * width - num_nonzeroes_act;
}

template <typename T>
int ComputeSemiSparseZeros(const Tensor* weights,
                           int* count_nonzeroes,
                           int* count_channels,
                           int* count_blocks,
                           int* flag_semi,
                           const int height,
                           const int width) {
  const T* data = weights->data<T>();
  int num_nonzeroes = 0;
  int num_nonzero_blocks2 = 0;
  int num_nonzero_blocks4 = 0;
  int align4 = height & (-4);
  int align2 = height & (-2);
  for (size_t oc = 0; oc < align4; oc += 4) {
    for (size_t ic = 0; ic < width; ic++) {
      const size_t row0_nonzero =
          static_cast<size_t>(data[oc * width + ic] != static_cast<T>(0));
      const size_t row1_nonzero =
          static_cast<size_t>(data[(oc + 1) * width + ic] != static_cast<T>(0));
      const size_t row2_nonzero =
          static_cast<size_t>(data[(oc + 2) * width + ic] != static_cast<T>(0));
      const size_t row3_nonzero =
          static_cast<size_t>(data[(oc + 3) * width + ic] != static_cast<T>(0));
      num_nonzeroes +=
          row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 +=
          (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 +=
          (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  for (size_t oc = align4; oc < align2; oc += 2) {
    for (size_t ic = 0; ic < width; ic++) {
      const size_t row0_nonzero =
          static_cast<size_t>(data[oc * width + ic] != static_cast<T>(0));
      const size_t row1_nonzero =
          static_cast<size_t>(data[(oc + 1) * width + ic] != static_cast<T>(0));
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = align2; oc < height; oc++) {
    for (size_t ic = 0; ic < width; ic++) {
      num_nonzeroes +=
          static_cast<size_t>(data[oc * width + ic] != static_cast<T>(0));
    }
  }
  *flag_semi = 0;
  *count_channels = height;
  *count_nonzeroes = num_nonzeroes;
  *count_blocks = num_nonzeroes;
  if ((num_block2_nonzeroes * 5 >= num_nonzero_blocks2 * 9) && (height > 1)) {
    // 2-channel blocks have 90%+ non-zeroes
    *count_channels = (*count_channels) / 2 + (*count_channels) % 2;
    // spmm_parameters = &xnn_params.f32.spmm2;
    *flag_semi = 1;
    // Non-zeroes which don't fit into whole 2-channel blocks, processed
    // one-by-one
    const size_t num_remaining_nonzeroes = num_nonzeroes - num_block2_nonzeroes;
    *count_nonzeroes = num_nonzero_blocks2 * 2 + num_remaining_nonzeroes;
    *count_blocks = num_nonzero_blocks2 + num_remaining_nonzeroes;
  }
  return height * width - (*count_nonzeroes);
}

template <typename T>
int ComputeSparseWeight(const Tensor* w_tensor,
                        const int M,
                        const int K,
                        const int N,
                        const int num_nonzeroes,
                        const int num_build_nonzeroes,
                        Tensor* nonzero_output_tensor,
                        Tensor* oc_nonzeros_tensor,
                        Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  std::vector<int32_t> act_diffs;
  act_diffs.resize(num_nonzeroes);
  int first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  for (int ocb = 0; ocb < M; ocb++) {
    oc_nonzeros[ocb] = 0;
    for (int ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          act_diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[ocb] += 1;
      }
    }
    if (oc_nonzeros[ocb] % 4 != 0) {
      int extra_zeros = 4 - (oc_nonzeros[ocb] % 4);
      for (int j = 0; j < extra_zeros; j++) {
        nonzero_output[nonzero_index++] = 0;
      }
    }
    if (ocb != 0) {
      int cur_rem = oc_nonzeros[ocb - 1] & 3;
      oc_nonzeros[ocb] =
          (cur_rem == 0)
              ? (oc_nonzeros[ocb] + oc_nonzeros[ocb - 1])
              : (oc_nonzeros[ocb] + oc_nonzeros[ocb - 1] + 4 - cur_rem);
    }
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    act_diffs[diff_index++] = diff * N;
  }
  int left_index = 0, right_index = 0;
  for (size_t ocb = 0; ocb < M; ocb++) {
    if (ocb == 0) {
      for (int i = 0; i < oc_nonzeros[ocb]; i++) {
        diffs[right_index++] = act_diffs[left_index++];
      }
      if (oc_nonzeros[ocb] % 4 != 0) {
        size_t extra_zeros = 4 - (oc_nonzeros[ocb] % 4);
        for (int j = 0; j < extra_zeros; j++) {
          diffs[right_index++] = 0;
        }
      }
    } else {
      int cur_rem = oc_nonzeros[ocb - 1] & 3;
      int cur_num =
          (cur_rem == 0)
              ? (oc_nonzeros[ocb] - oc_nonzeros[ocb - 1])
              : (oc_nonzeros[ocb] - (oc_nonzeros[ocb - 1] + 4 - cur_rem));
      for (int i = 0; i < cur_num; i++) {
        diffs[right_index++] = act_diffs[left_index++];
      }
      if (cur_num % 4 != 0) {
        size_t extra_zeros = 4 - (cur_num % 4);
        for (int j = 0; j < extra_zeros; j++) {
          diffs[right_index++] = 0;
        }
      }
    }
  }
  int tmp_diff = 0;
  int tmp_ik = 0;
  for (size_t ocb = 0; ocb < M; ocb++) {
    if (ocb == 0) {
      for (int ik = 0; ik < oc_nonzeros[ocb]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[ocb] - oc_nonzeros[ocb - 1]); ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
  }
  return first_ic;
}

template <typename T>
int ComputeSemiSparseWeight(const Tensor* w_tensor,
                            const int M,
                            const int K,
                            const int N,
                            const int count_nonzeroes,
                            const int count_channels,
                            const int count_blocks,
                            Tensor* nonzero_output_tensor,
                            Tensor* oc_nonzeros_tensor,
                            Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  int align2 = M & (-2);
  size_t output_channels_block_size = 2;
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  size_t block_index = 0, block_n = 0;
  for (size_t ocb = 0; ocb < align2; ocb += output_channels_block_size) {
    for (size_t ic = 0; ic < K; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |=
            (weights[(ocb + oco) * K + ic] != static_cast<T>(0));
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          nonzero_output[nonzero_index++] = weights[(ocb + oco) * K + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[block_index] += 1;
        block_n++;
      }
    }
    oc_nonzeros[block_index++] = block_n;
  }
  for (size_t ocb = align2; ocb < M; ocb++) {
    for (size_t ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[block_index] += 1;
        block_n++;
      }
    }
    oc_nonzeros[block_index++] = block_n;
  }
  int tmp_diff = 0;
  int tmp_ik = 0;
  size_t block_i = 0;
  for (size_t ocb = 0; ocb < align2; ocb += output_channels_block_size) {
    if (block_i == 0) {
      for (int ik = 0; ik < oc_nonzeros[block_i]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[block_i] - oc_nonzeros[block_i - 1]);
           ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
    block_i++;
  }
  for (size_t ocb = align2; ocb < M; ocb++) {
    if (block_i == 0) {
      for (int ik = 0; ik < oc_nonzeros[block_i]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[block_i] - oc_nonzeros[block_i - 1]);
           ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
    block_i++;
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    diffs[diff_index++] = diff * N;
  }
  return first_ic;
}

#ifdef LITE_WITH_ARM
bool test_spmm_fp32(bool tra,
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
                    bool has_semi,
                    int cls,
                    int ths,
                    float sparsity) {
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

  memcpy(dc_basic, dc, sizeof(float) * m * ldc);
  memcpy(dc_backup, dc, sizeof(float) * m * ldc);

  VLOG(4) << "spmm M: " << m << ", N: " << n << ", K: " << k
          << ", strides, lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
          << ", alpha: " << alpha << ", beta: " << beta
          << ", transA: " << (tra ? "true" : "false")
          << ", transB: " << (trb ? "true" : "false")
          << ", relu: " << (has_relu ? "true" : "false")
          << ", semi: " << (has_semi ? "true" : "false")
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
  zero_num = ComputeSemiSparseZeros<float>(&ta,
                                           &count_nonzeroes,
                                           &count_channels,
                                           &count_blocks,
                                           &f_semi,
                                           ch_out,
                                           ch_in);
  if (f_semi == 0) {
    zero_num =
        ComputeSparseZeros<float>(&ta, &num_build_nonzeroes, ch_out, ch_in);
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
    first_ic = ComputeSemiSparseWeight<float>(&ta,
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
    first_ic = ComputeSparseWeight<float>(&ta,
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

  const float* input = tb.data<float>();
  const float* nonzero_weights = nonzeros_output_t.data<float>();
  const int32_t* diffs = ic_diffs_t.data<int32_t>();
  const uint32_t* oc_nonzeros = oc_nonzeros_t.data<uint32_t>();
  const float* bias = has_bias ? tbias.data<float>() : nullptr;
  float* dout = tc.mutable_data<float>();
  const float* din = input + first_ic * im_size;
  int ic = k;
  int oc = m;
  paddle::lite::operators::SparseConvParam param;
  param.activation_param = act_param;
  for (int j = 0; j < FLAGS_warmup; ++j) {
    if (f_semi == 1) {
      paddle::lite::arm::math::sparse_semi_conv_fp32_pipelined(nonzero_weights,
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
      paddle::lite::arm::math::sparse_conv_fp32_pipelined(nonzero_weights,
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
      memcpy(dc, dc_backup, sizeof(float) * m * ldc);
    }
    t0.Start();
    if (f_semi == 1) {
      paddle::lite::arm::math::sparse_semi_conv_fp32_pipelined(nonzero_weights,
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
      paddle::lite::arm::math::sparse_conv_fp32_pipelined(nonzero_weights,
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
  return true;
}
#else
bool test_spmm_fp32(bool tra,
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
                    bool has_semi,
                    int cls,
                    int ths,
                    float sparsity) {
  return true;
}
#endif

TEST(TestSpmmF32, test_func_spmm_f32) {
  if (FLAGS_basic_test) {
#ifdef LITE_WITH_ARM
    paddle::lite::DeviceInfo::Init();
#endif
    LOG(INFO) << "run basic spmm test";
    for (auto& m : {1, 16, 64, 128}) {
      for (auto& n : {1, 32, 128, 256}) {
        for (auto& k : {1, 109, 512}) {
          for (auto& tra : {false}) {
            for (auto& trb : {false}) {
              for (auto& alpha : {1.f}) {
                for (auto& beta : {0.f}) {
                  for (auto& offset : {0}) {
                    for (auto& has_bias : {false, true}) {
                      for (auto& has_relu : {false, true}) {
                        for (auto& has_semi : {false, true}) {
                          for (auto& th : {1, 2, 4}) {
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
                              auto flag = test_spmm_fp32(tra,
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
                                                         has_semi,
                                                         FLAGS_power_mode,
                                                         th,
                                                         sp);
                              if (flag) {
                                VLOG(4)
                                    << "test m = " << m << ", n=" << n
                                    << ", k=" << k << ", bias: "
                                    << (has_bias ? "true" : "false")
                                    << ", relu: "
                                    << (has_relu ? "true" : "false")
                                    << ", semi: "
                                    << (has_semi ? "true" : "false")
                                    << ", trans A: " << (tra ? "true" : "false")
                                    << ", trans B: " << (trb ? "true" : "false")
                                    << " passed\n";
                              } else {
                                LOG(FATAL)
                                    << "test m = " << m << ", n=" << n
                                    << ", k=" << k << ", bias: "
                                    << (has_bias ? "true" : "false")
                                    << ", relu: "
                                    << (has_relu ? "true" : "false")
                                    << ", semi: "
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

TEST(TestSpmmF32Custom, test_func_spmm_f32_custom) {
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
  auto flag = test_spmm_fp32(FLAGS_traA,
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
                             FLAGS_flag_semi,
                             FLAGS_power_mode,
                             FLAGS_threads,
                             FLAGS_flag_sparsity);
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
