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

#include "lite/backends/host/math/inverse.h"
#include <cmath>

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T>
void partialLU(T *U, T *L, T *P, int n) {
  for (int j = 0; j < n - 1; j++) {
    int j_swap = j;
    for (int temp = j + 1; temp < n; temp++)
      if (std::abs(U[temp * n + j]) > std::abs(U[j_swap * n + j]))
        j_swap = temp;
    CHECK_GT(std::abs(U[j_swap * n + j]), 0)
        << "the input matrix is not invertible";
    if (j_swap != j) {
      for (int temp = j; temp < n; temp++) {
        T swap_temp = U[j_swap * n + temp];
        U[j_swap * n + temp] = U[j * n + temp];
        U[j * n + temp] = swap_temp;
      }
      for (int temp = 0; temp < j; temp++) {
        T swap_temp = L[j_swap * n + temp];
        L[j_swap * n + temp] = L[j * n + temp];
        L[j * n + temp] = swap_temp;
      }
      for (int temp = 0; temp < n; temp++) {
        T swap_temp = P[j_swap * n + temp];
        P[j_swap * n + temp] = P[j * n + temp];
        P[j * n + temp] = swap_temp;
      }
    }
    for (int i = j + 1; i < n; i++) {
      L[i * n + j] = U[i * n + j] / U[j * n + j];
      for (int k = j; k < n; k++) U[i * n + k] -= L[i * n + j] * U[j * n + k];
    }
  }
}

template <typename T>
void LowInverse(T *L, int n) {
  for (int j = 0; j < n; j++) {
    for (int i = j + 1; i < n; i++) {
      T temp = 0;
      for (int k = j; k <= i - 1; k++) temp += L[i * n + k] * L[k * n + j];
      L[i * n + j] = -temp;
    }
  }
}

template <typename T>
void UpperInverse(T *U, int n) {
  for (int j = n - 1; j >= 0; j--) {
    U[j * n + j] = 1 / U[j * n + j];
    for (int i = j - 1; i >= 0; i--) {
      T temp = 0;
      for (int k = i + 1; k <= j; k++) temp += (U[i * n + k] * U[k * n + j]);
      U[i * n + j] = -temp / U[i * n + i];
    }
  }
}

template <typename T>
void MatMul(T *U_1, T *L_1, T *P, int n, T *out) {
  T *temp_array =
      reinterpret_cast<T *>(TargetMalloc(TARGET(kHost), sizeof(T) * n * n));
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      T temp = 0;
      for (int k = 0; k < n; k++) temp += U_1[i * n + k] * L_1[k * n + j];
      temp_array[i * n + j] = temp;
    }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      T temp = 0;
      for (int k = 0; k < n; k++) temp += temp_array[i * n + k] * P[k * n + j];
      out[i * n + j] = temp;
    }
  TargetFree(TARGET(kHost), temp_array);
}

template <typename InType>
void inverse_func(const lite::Tensor *input, lite::Tensor *output) {
  auto input_ddim = input->dims();
  auto output_ddim = output->dims();
  const int rank = input_ddim.size();
  const int batch_size = input_ddim.count(0, rank - 2);
  int n = input_ddim[rank - 1];
  const InType *in_ptr = input->data<InType>();
  InType *out_ptr = output->mutable_data<InType>();

  InType *L = reinterpret_cast<InType *>(
      TargetMalloc(TARGET(kHost), sizeof(InType) * n * n));
  InType *U = reinterpret_cast<InType *>(
      TargetMalloc(TARGET(kHost), sizeof(InType) * n * n));
  InType *P = reinterpret_cast<InType *>(
      TargetMalloc(TARGET(kHost), sizeof(InType) * n * n));

  for (int i = 0; i < batch_size; i++) {
    memset(P, 0, sizeof(InType) * n * n);
    for (int i = 0; i < n; i++) P[i * n + i] = 1;
    memcpy(L, P, sizeof(InType) * n * n);
    memcpy(U, in_ptr + i * n * n, sizeof(InType) * n * n);
    partialLU(U, L, P, n);
    LowInverse(L, n);
    UpperInverse(U, n);
    MatMul(U, L, P, n, out_ptr + i * n * n);
  }
  TargetFree(TARGET(kHost), L);
  TargetFree(TARGET(kHost), U);
  TargetFree(TARGET(kHost), P);
}

template void inverse_func<float>(const lite::Tensor *input,
                                  lite::Tensor *output);

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
