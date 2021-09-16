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
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/timer.h"

// for eval
DEFINE_string(model_dir, "", "model dir");
#ifdef LITE_WITH_METAL
DEFINE_string(metal_dir, "", "metal lib dir");
#endif
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_int32(power_mode,
             3,
             "arm power mode: "
             "0 for big cluster, "
             "1 for little cluster, "
             "2 for all cores, "
             "3 for no bind");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(im_width, 224, "image width");
DEFINE_int32(im_height, 224, "image height");
DEFINE_bool(int8, false, "is run int8");

namespace paddle {
namespace lite {

inline double GetCurrentUS() {
  return static_cast<double>(lite::Timer::GetCurrentUS());
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

void ReadTxtFile(const std::string& file_path, float* dest, int num) {
  CHECK(!file_path.empty());
  CHECK(dest != nullptr);
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    LOG(FATAL) << "open file error:" << file_path;
  }
  for (int i = 0; i < num; i++) {
    ifs >> dest[i];
  }
  ifs.close();
}

template <typename T>
T ShapeProduction(const std::vector<T>& shape) {
  T num = 1;
  for (auto i : shape) {
    num *= i;
  }
  return num;
}

template <class T>
void FillTensor(
    const std::shared_ptr<paddle::lite_api::PaddlePredictor>& predictor,
    int tensor_id,
    const std::vector<int64_t>& tensor_shape,
    const std::vector<T>& tensor_value,
    const std::vector<std::vector<uint64_t>> tensor_lod = {}) {
  auto tensor_x = predictor->GetInput(tensor_id);
  tensor_x->Resize(tensor_shape);
  int64_t tensor_size = 1;
  for (size_t i = 0; i < tensor_shape.size(); i++) {
    tensor_size *= tensor_shape[i];
  }
  CHECK_EQ(static_cast<size_t>(tensor_size), tensor_value.size());
  memcpy(tensor_x->mutable_data<T>(),
         tensor_value.data(),
         sizeof(T) * tensor_size);
  if (!tensor_lod.empty()) {
    tensor_x->SetLoD(tensor_lod);
  }
}

}  // namespace lite
}  // namespace paddle
