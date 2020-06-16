// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include "compute_api.h"  // NOLINT
#include "paddle_api.h"   // NOLINT

using namespace paddle::lite_api;  // NOLINT

template <typename Dtype>
void fill_tensor_host_const_impl(Dtype* dio, Dtype value, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = value;
  }
}

int64_t dim_production(const Tensor& t) {
  shape_t s = t.shape();
  int64_t n = 1;
  for (int i = 0; i < s.size(); ++i) {
    n *= s[i];
  }
  return n;
}
/**
 *  \brief Fill the host tensor buffer with rand value.
 *  \param tensor  The reference of input tensor.
 */
void fill_tensor_const(Tensor& tensor, float value) {  // NOLINT
  int64_t size = dim_production(tensor);
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      fill_tensor_host_const_impl(
          tensor.mutable_data<int8_t>(), static_cast<signed char>(value), size);
      break;
    case PRECISION(kInt32):
      fill_tensor_host_const_impl(
          tensor.mutable_data<int>(), static_cast<int>(value), size);
      break;
    case PRECISION(kFloat):
      fill_tensor_host_const_impl(
          tensor.mutable_data<float>(), static_cast<float>(value), size);
      break;
    default:
      std::cerr << "data type is unsupported now." << std::endl;
      assert(0);
  }
}

template <typename Dtype>
void fill_tensor_host_rand_impl(Dtype* dio, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    Dtype rand_x = static_cast<Dtype>(rand() % 256);  // NOLINT
    dio[i] = (rand_x - 128) / 128;
  }
}

template <>
void fill_tensor_host_rand_impl<signed char>(signed char* dio, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = rand() % 256 - 128;  // NOLINT
  }
}
template <>
void fill_tensor_host_rand_impl<unsigned char>(unsigned char* dio,
                                               int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = rand() % 256;  // NOLINT
  }
}
/**
 *  \brief Fill the host tensor buffer with rand value.
 *  \param The reference of input tensor.
 */
void fill_tensor_rand(Tensor& tensor) {  // NOLINT
  int64_t size = dim_production(tensor);
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      fill_tensor_host_rand_impl(tensor.mutable_data<int8_t>(), size);
      break;
    case PRECISION(kInt32):
      fill_tensor_host_rand_impl(tensor.mutable_data<int>(), size);
      break;
    case PRECISION(kFloat):
      fill_tensor_host_rand_impl(tensor.mutable_data<float>(), size);
      break;
    default:
      std::cerr << "data type: is unsupported now" << std::endl;
      assert(0);
  }
}

template <typename Dtype>
void fill_tensor_host_rand_impl2(Dtype* dio,
                                 Dtype vstart,
                                 Dtype vend,
                                 int64_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (int64_t i = 0; i < size; ++i) {
    Dtype random_num = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
    dio[i] = random_num;
  }
}

/**
 *  \brief Fill the host tensor buffer with rand value from vstart to vend.
 *  \param tensor The reference of input tensor.
 */
void fill_tensor_rand(Tensor& tensor, float vstart, float vend) {  // NOLINT
  int64_t size = dim_production(tensor);
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      fill_tensor_host_rand_impl2(tensor.mutable_data<int8_t>(),
                                  static_cast<signed char>(vstart),
                                  static_cast<signed char>(vend),
                                  size);
      break;
    case PRECISION(kInt32):
      fill_tensor_host_rand_impl2(tensor.mutable_data<int>(),
                                  static_cast<int>(vstart),
                                  static_cast<int>(vend),
                                  size);
      break;
    case PRECISION(kFloat):
      fill_tensor_host_rand_impl2(
          tensor.mutable_data<float>(), vstart, vend, size);
      break;
    default:
      std::cerr << "data type: is unsupported now" << std::endl;
      assert(0);
  }
}

template <typename Dtype>
void print_tensor_host_impl(const Dtype* din, int64_t size, int64_t width);

template <>
void print_tensor_host_impl(const float* din, int64_t size, int64_t width) {
  for (int i = 0; i < size; ++i) {
    printf("%.6f ", din[i]);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <>
void print_tensor_host_impl(const int* din, int64_t size, int64_t width) {
  for (int i = 0; i < size; ++i) {
    printf("%d ", din[i]);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

template <>
void print_tensor_host_impl(const signed char* din,
                            int64_t size,
                            int64_t width) {
  for (int i = 0; i < size; ++i) {
    printf("%d ", din[i]);
    if ((i + 1) % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
}
/**
 *  \brief Print the data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
void print_tensor(const Tensor& tensor) {
  printf("host tensor data size: %ld\n", dim_production(tensor));
  int64_t size = dim_production(tensor);
  int64_t width = tensor.shape()[tensor.shape().size() - 1];
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      print_tensor_host_impl(tensor.data<int8_t>(), size, width);
      break;
    case PRECISION(kInt32):
      print_tensor_host_impl(tensor.data<int>(), size, width);
      break;
    case PRECISION(kFloat):
      print_tensor_host_impl(tensor.data<float>(), size, width);
      break;
    default:
      std::cerr << "data type: is unsupported now" << std::endl;
      assert(0);
  }
}

template <typename Dtype>
double tensor_mean_value_host_impl(const Dtype* din, int64_t size) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    sum += din[i];
  }
  return sum / size;
}

double tensor_mean(const Tensor& tensor) {
  int64_t size = dim_production(tensor);
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      return tensor_mean_value_host_impl(tensor.data<int8_t>(), size);
    case PRECISION(kInt32):
      return tensor_mean_value_host_impl(tensor.data<int>(), size);
    case PRECISION(kFloat):
      return tensor_mean_value_host_impl(tensor.data<float>(), size);
    default:
      std::cerr << "data type: is unsupported now" << std::endl;
      assert(0);
  }
  return 0.0;
}

template <typename dtype>
void data_diff_kernel(const dtype* src1_truth,
                      const dtype* src2,
                      int size,
                      double& max_ratio,   // NOLINT
                      double& max_diff) {  // NOLINT
  const double eps = 1e-6f;
  max_diff = fabs(src1_truth[0] - src2[0]);
  max_ratio = fabs(max_diff) / (std::abs(src1_truth[0]) + eps);
  for (int i = 1; i < size; ++i) {
    double diff = fabs(src1_truth[i] - src2[i]);
    double ratio = fabs(diff) / (std::abs(src1_truth[i]) + eps);
    if (max_ratio < ratio) {
      max_diff = diff;
      max_ratio = ratio;
    }
  }
}

void tensor_cmp_host(const Tensor& src1_basic,
                     const Tensor& src2,
                     double& max_ratio,   // NOLINT
                     double& max_diff) {  // NOLINT
  max_ratio = 0.;
  max_diff = 0.;
  int64_t size = dim_production(src1_basic);
  int64_t size2 = dim_production(src2);
  if (size != size2) {
    std::cerr << "ERROR: tensor_cmp_host: wrong shape" << std::endl;
    assert(0);
  }
  auto ptype1 = src1_basic.precision();
  auto ptype2 = src2.precision();
  if (ptype1 != ptype2) {
    std::cerr << "ERROR: tensor_cmp_host: wrong data type" << std::endl;
    assert(0);
  }
  if (size == 0) return;
  switch (src1_basic.precision()) {
    case PRECISION(kFloat):
      data_diff_kernel(src1_basic.data<float>(),
                       src2.data<float>(),
                       size,
                       max_ratio,
                       max_diff);
      return;
    case PRECISION(kInt32):
      data_diff_kernel(
          src1_basic.data<int>(), src2.data<int>(), size, max_ratio, max_diff);
      return;
    case PRECISION(kInt8):
      data_diff_kernel(src1_basic.data<int8_t>(),
                       src2.data<int8_t>(),
                       size,
                       max_ratio,
                       max_diff);
      return;
    default:
      std::cerr << "data type: is unsupported now" << std::endl;
      assert(0);
  }
}

template <typename dtype>
void tensor_diff_kernel(const dtype* src1,
                        const dtype* src2,
                        dtype* dst,
                        int64_t size) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src1[i] - src2[i];
  }
}
void tensor_diff(const Tensor& t1, const Tensor& t2, Tensor& tdiff) {  // NOLINT
  int64_t size1 = dim_production(t1);
  int64_t size2 = dim_production(t2);
  if (size1 != size2) {
    std::cerr << "ERROR: tensor_diff: wrong shape" << std::endl;
    assert(0);
  }
  auto ptype1 = t1.precision();
  auto ptype2 = t2.precision();
  if (ptype1 != ptype2) {
    std::cerr << "ERROR: tensor_diff: wrong data type" << std::endl;
    assert(0);
  }
  tdiff.Resize(t1.shape());
  switch (t1.precision()) {
    case PRECISION(kFloat):
      tensor_diff_kernel(t1.data<float>(),
                         t2.data<float>(),
                         tdiff.mutable_data<float>(),
                         size1);
      return;
    case PRECISION(kInt32):
      tensor_diff_kernel(
          t1.data<int>(), t2.data<int>(), tdiff.mutable_data<int>(), size1);
    case PRECISION(kInt8):
      tensor_diff_kernel(t1.data<int8_t>(),
                         t2.data<int8_t>(),
                         tdiff.mutable_data<int8_t>(),
                         size1);
      return;
    default:
      std::cerr << "data type: is unsupported now" << std::endl;
      assert(0);
  }
}

template <typename T>
class TimeList {
 public:
  void Clear() { laps_t_.clear(); }
  void Add(T t) { laps_t_.push_back(t); }
  T Last(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return laps_t_.back();
  }
  T Max(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return *std::max_element((laps_t_.begin() + offset), laps_t_.end());
  }
  T Min(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return *std::min_element((laps_t_.begin() + offset), laps_t_.end());
  }
  T Sum(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return std::accumulate((laps_t_.begin() + offset), laps_t_.end(), 0.0);
  }
  size_t Size(size_t offset = 0) const {
    size_t size = (laps_t_.size() <= offset) ? 0 : (laps_t_.size() - offset);
    return size;
  }
  T Avg(size_t offset = 0) const {
    if (!Size(offset)) {
      return 0;
    }
    return Sum(offset) / Size(offset);
  }
  const std::vector<T>& Raw() const { return laps_t_; }

 private:
  std::vector<T> laps_t_;
};

class Timer {
 public:
  Timer() = default;
  virtual ~Timer() = default;

  void Reset() { laps_t_.Clear(); }
  void Start() { t_start_ = std::chrono::system_clock::now(); }
  float Stop() {
    t_stop_ = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t_stop_ -
                                                                    t_start_);
    float elapse_ms = 1000.f * static_cast<float>(ts.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den;
    this->laps_t_.Add(elapse_ms);
    return elapse_ms;
  }
  float AvgLapTimeMs() const { return laps_t_.Avg(); }
  const TimeList<float>& LapTimes() const { return laps_t_; }

 protected:
  TimeList<float> laps_t_;

 private:
  std::chrono::time_point<std::chrono::system_clock> t_start_, t_stop_;
};
