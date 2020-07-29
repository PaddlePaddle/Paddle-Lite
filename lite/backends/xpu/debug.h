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

#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace xpu {

template <typename T>
void DumpCPUMem(const T* ptr,
                size_t len,
                const std::string& comment = "",
                size_t stride = 1,
                size_t item_per_line = 30) {
  size_t after_stride_len = (len + stride - 1) / stride;
  std::unique_ptr<T[]> after_stride(new T[after_stride_len]);
  for (size_t i = 0; i < after_stride_len; ++i) {
    after_stride[i] = ptr[i * stride];
  }
  double sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += ptr[i];
  }

  printf(
      "------------------------------ [%s] len=%zd stride=%zd sum=%f BEGIN "
      "------------------------------\n",
      comment.c_str(),
      len,
      stride,
      sum);
  size_t nline = (after_stride_len + item_per_line - 1) / item_per_line;
  for (size_t i = 0; i < nline; ++i) {
    size_t line_begin = i * item_per_line;
    size_t line_end = line_begin + item_per_line;
    printf("line[%04zd] -- ", i);
    for (size_t ii = line_begin; (ii < line_end) && (ii < after_stride_len);
         ++ii) {
      if (std::is_same<T, float>::value) {
        printf("%.6f, ", static_cast<float>(after_stride[ii]));
      } else if (std::is_same<T, int16_t>::value) {
        printf("%d ", static_cast<int>(after_stride[ii]));
      } else {
        // CHECK(false) << "unknown type";
      }
    }
    printf("\n");
  }
  printf(
      "------------------------------ [%s] len=%zd stride=%zd sum=%f  END  "
      "------------------------------\n",
      comment.c_str(),
      len,
      stride,
      sum);
}

template <typename T>
void DumpXPUMem(const T* ptr,
                size_t len,
                const std::string& comment = "",
                size_t stride = 1,
                size_t item_per_line = 30) {
  size_t after_stride_len = (len + stride - 1) / stride;
  std::unique_ptr<T[]> cpu_mem(new T[len]);
  XPU_CALL(xpu_memcpy(
      cpu_mem.get(), ptr, len * sizeof(T), XPUMemcpyKind::XPU_DEVICE_TO_HOST));
  std::unique_ptr<T[]> after_stride(new T[after_stride_len]);
  for (size_t i = 0; i < after_stride_len; ++i) {
    after_stride[i] = cpu_mem[i * stride];
  }
  double sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += cpu_mem[i];
  }

  printf(
      "------------------------------ [%s] len=%zd stride=%zd sum=%f BEGIN "
      "------------------------------\n",
      comment.c_str(),
      len,
      stride,
      sum);
  size_t nline = (after_stride_len + item_per_line - 1) / item_per_line;
  for (size_t i = 0; i < nline; ++i) {
    size_t line_begin = i * item_per_line;
    size_t line_end = line_begin + item_per_line;
    printf("line[%04zd] -- ", i);
    for (size_t ii = line_begin; (ii < line_end) && (ii < after_stride_len);
         ++ii) {
      if (std::is_same<T, float>::value) {
        printf("%.6f, ", static_cast<float>(after_stride[ii]));
      } else if (std::is_same<T, int16_t>::value) {
        printf("%d ", static_cast<int>(after_stride[ii]));
      } else {
        // CHECK(false) << "unknown type";
      }
    }
    printf("\n");
  }
  printf(
      "------------------------------ [%s] len=%zd stride=%zd sum=%f  END  "
      "------------------------------\n",
      comment.c_str(),
      len,
      stride,
      sum);
}

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
