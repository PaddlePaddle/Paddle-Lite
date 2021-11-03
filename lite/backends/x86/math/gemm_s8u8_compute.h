/* Copyright (c) 2021 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string.h>
#include <algorithm>
#include <cmath>
#include "lite/backends/x86/math/gemm_s8u8_kernel.h"
#include "lite/backends/x86/math/gemm_s8u8_pack.h"
#include "lite/core/memory.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

#define PARAM_INIT               \
  _is_trans_A = is_trans_A;      \
  _is_trans_B = is_trans_B;      \
  _M = M;                        \
  _N = N;                        \
  _K = K;                        \
  _A = A;                        \
  _ldc = ldc;                    \
  _Sa = const_cast<float *>(Sa); \
  _Sb = Sb;                      \
  _Sc = Sc;                      \
  _relu_type = relu_type;        \
  _relu_alpha = relu_alpha;

template <typename TYPE_C>
class generate_gemm_s8u8_x86_kern {
 public:
  explicit generate_gemm_s8u8_x86_kern(bool is_trans_A,
                                       bool is_trans_B,
                                       int M,
                                       int N,
                                       int K,
                                       const int8_t *A,
                                       int ldc,
                                       const float *Sa,
                                       const float Sb,
                                       const float Sc,
                                       const float *bias,
                                       int relu_type,
                                       float relu_alpha) {
    PARAM_INIT
    gemm_int8_init(M, N, K, bias);
  }

  ~generate_gemm_s8u8_x86_kern() { gemm_int8_deinit(); }

  void compute(const int8_t *A, const int8_t *B, TYPE_C *C) {
    if (_relu_type < 0 || _relu_type > 3) {
      LOG(FATAL) << "relu_type: 1 for relu, 2 for relu6, 3 for leakyrelu, but "
                    "receive is "
                 << _relu_type;
    }

    _B = B;
    _C = C;
    int loop_m, loop_n;
    int block_m, block_n;
    int min_m, min_n;
    int8_t *cur_a = _pack_A;
    const int8_t *cur_b = _B;
    TYPE_C *cur_c = _C;
    calc_block(_M, _N, _K, &block_m, &block_n);
    for (loop_n = 0; loop_n < _N; loop_n += block_n) {
      min_n = ((_N - loop_n) >= block_n) ? block_n : (_N - loop_n);
      cur_b = _is_trans_B ? (_B + loop_n * _K) : (_B + loop_n);
      int step = _is_trans_B ? _K : _N;
      packB_i82u8(min_n, _K, step, cur_b, _pack_B, _is_trans_B);

      for (loop_m = 0; loop_m < _M; loop_m += block_m) {
        min_m = ((_M - loop_m) >= block_m) ? block_m : (_M - loop_m);
        cur_a = _pack_A + loop_m * _k_align4;
        cur_c = _C + loop_m * _ldc + loop_n;

        // kernel
        gemm_kernel_loop_int8(min_m,
                              min_n,
                              _K,
                              cur_a,
                              _pack_B,
                              cur_c,
                              _ldc,
                              _scale + loop_m,
                              _re_bias + loop_m,
                              _relu_type,
                              _relu_alpha);
      }
    }
  }

 private:
  // inner param
  int _k_align4;
  int _relu_type;
  int _M, _N, _K, _ldc;
  float _Sb, _Sc;
  float _relu_alpha;
  bool _C_is_int8;
  bool _is_trans_A;
  bool _is_trans_B;
  // divide block param
  const int _unroll_n = 32;
  const int _unroll_m = 2;
  const int _l2_size = 262144;  // 256K
  // work buffer
  TYPE_C *_C{nullptr};
  float *_Sa{nullptr};
  float *_scale{nullptr};
  float *_in_bias{nullptr};
  float *_re_bias{nullptr};
  int8_t *_pack_A{nullptr};
  uint8_t *_pack_B{nullptr};
  const int8_t *_A{nullptr};
  const int8_t *_B{nullptr};

  // prepare input data
  void repack_bias(bool is_trans,
                   int M,
                   int K,
                   const float *bias,
                   float *out,
                   float *Sa,
                   float Sb,
                   float Sc,
                   const int8_t *A);

  void calc_scale(int M, float *Sa, float Sb, float Sc, float *out);

  // pack A
  void prepackA_i8(
      int M, int K, const int8_t *A, int8_t *pack_A, bool is_trans) {
    memset(pack_A, 0, _M * _k_align4);  // important, can't delete
    gemm_s8u8s8_prepackA(M, K, A, pack_A, is_trans);
  }

  // pack B
  void packB_i82u8(int N,
                   int K,
                   int stride,
                   const int8_t *B,
                   uint8_t *pack_B,
                   bool is_trans) {
    gemm_s8u8s8_runpackB(N, K, stride, B, pack_B, is_trans);
  }

  void gemm_int8_init(int M, int N, int K, const float *bias) {
    int K_align4 = (K + 3) >> 2;
    int block_n = 0;
    int block_m = 0;
    // calc block according to L2 size
    K_align4 = K_align4 << 2;
    _k_align4 = K_align4;
    calc_block(M, N, K, &block_m, &block_n);
    // malloc work_buf
    _pack_A = reinterpret_cast<int8_t *>(
        TargetMalloc(TARGET(kX86), block_m * K_align4));
    _pack_B = reinterpret_cast<uint8_t *>(
        TargetMalloc(TARGET(kX86), block_n * K_align4));
    _re_bias = reinterpret_cast<float *>(
        TargetMalloc(TARGET(kX86), M * sizeof(float)));
    _scale = reinterpret_cast<float *>(
        TargetMalloc(TARGET(kX86), M * sizeof(float)));
    // if no bias, malloc a buffer and set all zero.
    if (bias == nullptr) {
      _in_bias = reinterpret_cast<float *>(
          TargetMalloc(TARGET(kX86), M * sizeof(float)));
      memset(_in_bias, 0, M * sizeof(float));
      repack_bias(_is_trans_A, M, K, _in_bias, _re_bias, _Sa, _Sb, _Sc, _A);
    } else {
      repack_bias(_is_trans_A, M, K, bias, _re_bias, _Sa, _Sb, _Sc, _A);
    }
    calc_scale(M, _Sa, _Sb, _Sc, _scale);
    prepackA_i8(M, K, _A, _pack_A, _is_trans_A);
  }

  void gemm_int8_deinit() {
    if (_pack_A != nullptr) {
      TargetFree(TARGET(kX86), _pack_A);
    }
    if (_pack_B != nullptr) {
      TargetFree(TARGET(kX86), _pack_B);
    }
    if (_re_bias != nullptr) {
      TargetFree(TARGET(kX86), _re_bias);
    }
    if (_scale != nullptr) {
      TargetFree(TARGET(kX86), _scale);
    }
    if (_in_bias != nullptr) {
      TargetFree(TARGET(kX86), _in_bias);
    }
  }

  void calc_block(int M, int N, int K, int *blk_m, int *blk_n);
};

#undef PARAM_INIT

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
