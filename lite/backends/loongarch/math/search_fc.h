/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "lite/backends/loongarch/fluid/data_type.h"
#include "lite/backends/loongarch/math/blas.h"
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template <typename DeviceContext, typename T>
void call_gemm(const BlasT<lite::TargetType::kLoongArch, T> blas,
               const CBLAS_TRANSPOSE TransA,
               const CBLAS_TRANSPOSE TransB,
               const int M,
               const int N,
               const int K,
               const T alpha,
               const T* A,
               const T* B,
               const T beta,
               T* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

static const unsigned int LASX_STEP_SIZE = 8;
static const unsigned int LSX_STEP_SIZE = 4;
static const unsigned int LASX_CUT_LEN_MASK = 7U;
static const unsigned int LSX_CUT_LEN_MASK = 3U;

template <typename T>
inline void vector_eltadd(const T* x, const T* y, T* z, size_t len) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(__loongarch_asx)
  lll = len & ~LASX_CUT_LEN_MASK;
  for (jjj = 0; jjj < lll; jjj += LASX_STEP_SIZE) {
    lasx_storeu_f32(
        z + jjj,
        lasx_add_f32(lasx_loadu_f32(x + jjj), lasx_loadu_f32(y + jjj)));
  }
#elif defined(__loongarch_sx)
  lll = len & ~LSX_CUT_LEN_MASK;

  for (jjj = 0; jjj < lll; jjj += LSX_STEP_SIZE) {
    lsx_storeu_f32(z + jjj,
                 lsx_add_f32(lsx_loadu_f32(x + jjj), lsx_loadu_f32(y + jjj)));
  }
#endif
  for (; jjj < len; jjj++) {
    z[jjj] = x[jjj] + y[jjj];
  }
}

template <lite::TargetType Target, typename T>
class SearchFcFunctor {
 public:
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& X,
                  const lite::Tensor& W,
                  const lite::Tensor& b,
                  lite::Tensor* Out,
                  int out_size);
};

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle

#define FOR_ALL_TYPES(macro) macro(float);
