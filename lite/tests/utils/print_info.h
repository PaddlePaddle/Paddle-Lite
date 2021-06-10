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
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <string>
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;

void print_gops_info(std::string op_name,
                     const DDim dim_in,
                     const DDim dim_out,
                     Timer t0,
                     double gops) {
  VLOG(4) << op_name << ": input shape: " << dim_in << ", output shape"
          << dim_out << ", running time, avg: " << t0.LapTimes().Avg()
          << ", min time: " << t0.LapTimes().Min()
          << ", total GOPS: " << 1e-9 * gops
          << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
          << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();
}

void print_diff_info(double max_diff, double max_ratio) {
  VLOG(4) << "compare result, max diff: " << max_diff
          << ", max ratio: " << max_ratio;
}

void print_tensor_info_common(Tensor src,
                              Tensor basic,
                              Tensor lite,
                              bool has_din) {
  if (has_din) {
    LOG(WARNING) << "din";
    print_tensor(src);
  }
  LOG(WARNING) << "basic result";
  print_tensor(basic);
  LOG(WARNING) << "lite result";
  print_tensor(lite);
  Tensor tdiff;
  tdiff.Resize(basic.dims());
  tdiff.set_precision(PRECISION(kFloat));
  tensor_diff(basic, lite, tdiff);
  LOG(WARNING) << "diff result";
  print_tensor(tdiff);
}

#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
inline void data_diff(const float16_t* src1_truth,
                      const float16_t* src2,
                      float16_t* dst,
                      int size,
                      double& max_ratio,   // NOLINT
                      double& max_diff) {  // NOLINT
  const double eps = 1e-6f;
  max_diff = fabs(src1_truth[0] - src2[0]);
  dst[0] = max_diff;
  max_ratio = fabs(max_diff) / (std::abs(src1_truth[0]) + eps);
  for (int i = 1; i < size; ++i) {
    double diff = fabs(src1_truth[i] - src2[i]);
    dst[i] = diff;
    max_diff = max_diff < diff ? diff : max_diff;
    double ratio = fabs(diff) / (std::abs(src1_truth[i]) + eps);
    if (max_ratio < ratio) {
      max_ratio = ratio;
    }
  }
}

inline void print_data_fp16(const float16_t* din, int64_t size, int64_t width) {
  for (int i = 0; i < size; ++i) {
    printf("%.6f ", din[i]);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

void print_tensor_info_fp16(const float16_t* basic_ptr,
                            const float16_t* saber_ptr,
                            const float16_t* diff_ptr,
                            int size,
                            int width) {
  LOG(WARNING) << "basic result";
  print_data_fp16(basic_ptr, size, width);
  LOG(WARNING) << "lite result";
  print_data_fp16(saber_ptr, size, width);
  LOG(WARNING) << "diff result";
  print_data_fp16(diff_ptr, size, width);
}
#endif
