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

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif  // _WIN32

#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
#endif
template <typename Dtype>
void fill_tensor_host_const_impl(Dtype* dio, Dtype value, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = value;
  }
}
/**
 *  \brief Fill the host tensor buffer with rand value.
 *  \param tensor  The reference of input tensor.
 */
void fill_tensor_const(Tensor& tensor, float value) {  // NOLINT
  int64_t size = tensor.numel();
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      fill_tensor_host_const_impl(
          tensor.mutable_data<int8_t>(), static_cast<signed char>(value), size);
      break;
    case PRECISION(kInt16):
      fill_tensor_host_const_impl(
          tensor.mutable_data<int16_t>(), static_cast<int16_t>(value), size);
      break;
    case PRECISION(kInt32):
      fill_tensor_host_const_impl(
          tensor.mutable_data<int>(), static_cast<int>(value), size);
      break;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      fill_tensor_host_const_impl(tensor.mutable_data<float16_t>(),
                                  static_cast<float16_t>(value),
                                  size);
      break;
#endif
    case PRECISION(kFloat):
      fill_tensor_host_const_impl(
          tensor.mutable_data<float>(), static_cast<float>(value), size);
      break;
    default:
      LOG(FATAL) << "data type: " << PrecisionRepr(type)
                 << " is unsupported now";
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
void fill_tensor_host_rand_impl<int16_t>(int16_t* dio, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    dio[i] = (rand() % 256 - 128) * 2;  // NOLINT
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
  int64_t size = tensor.numel();
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      fill_tensor_host_rand_impl(tensor.mutable_data<int8_t>(), size);
      break;
    case PRECISION(kInt16):
      fill_tensor_host_rand_impl(tensor.mutable_data<int16_t>(), size);
      break;
    case PRECISION(kInt32):
      fill_tensor_host_rand_impl(tensor.mutable_data<int>(), size);
      break;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      fill_tensor_host_rand_impl(tensor.mutable_data<float16_t>(), size);
      break;
#endif
    case PRECISION(kFloat):
      fill_tensor_host_rand_impl(tensor.mutable_data<float>(), size);
      break;
    default:
      LOG(FATAL) << "data type: " << PrecisionRepr(type)
                 << " is unsupported now";
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
    // dio[i] = i % 110 +1;
  }
}

/**
 *  \brief Fill the host tensor buffer with rand value from vstart to vend.
 *  \param tensor The reference of input tensor.
 */
void fill_tensor_rand(Tensor& tensor, float vstart, float vend) {  // NOLINT
  int64_t size = tensor.numel();
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
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      fill_tensor_host_rand_impl2(tensor.mutable_data<float16_t>(),
                                  static_cast<float16_t>(vstart),
                                  static_cast<float16_t>(vend),
                                  size);
      break;
#endif
    case PRECISION(kFloat):
      fill_tensor_host_rand_impl2(
          tensor.mutable_data<float>(), vstart, vend, size);
      break;
    default:
      LOG(FATAL) << "data type: " << PrecisionRepr(type)
                 << " is unsupported now";
  }
}

template <typename Dtype>
void print_tensor_host_impl(const Dtype* din, int64_t size, int64_t width) {
  std::ostringstream os;
  for (int i = 0; i < size; ++i) {
    os << din[i] << " ";
    if ((i + 1) % width == 0) {
      VLOG(4) << os.str();
      os.str("");
    }
  }
  VLOG(4) << "\n";
}

/**
 *  \brief Print the data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
void print_tensor(const Tensor& tensor) {
  printf("host tensor data size: %ld\n", tensor.numel());
  int64_t size = tensor.numel();
  int64_t width = tensor.dims()[tensor.dims().size() - 1];
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
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      print_tensor_host_impl(tensor.data<float16_t>(), size, width);
      break;
#endif
    default:
      LOG(FATAL) << "data type: " << PrecisionRepr(type)
                 << " is unsupported now";
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
  int64_t size = tensor.numel();
  PrecisionType type = tensor.precision();
  switch (type) {
    case PRECISION(kInt8):
      return tensor_mean_value_host_impl(tensor.data<int8_t>(), size);
    case PRECISION(kInt32):
      return tensor_mean_value_host_impl(tensor.data<int>(), size);
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      return tensor_mean_value_host_impl(tensor.data<float16_t>(), size);
#endif
    case PRECISION(kFloat):
      return tensor_mean_value_host_impl(tensor.data<float>(), size);
    default:
      LOG(FATAL) << "data type: " << PrecisionRepr(type)
                 << " is unsupported now";
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
  int64_t size = src1_basic.numel();
  CHECK_EQ(size, src2.numel()) << "ERROR: tensor_cmp_host: wrong shape";
  auto ptype1 = PrecisionRepr(src1_basic.precision());
  auto ptype2 = PrecisionRepr(src2.precision());
  CHECK_EQ(ptype1, ptype2) << "ERROR: tensor_cmp_host: wrong data type";
  if (size == 0) return;
  switch (src1_basic.precision()) {
    case PRECISION(kFloat):
      data_diff_kernel(src1_basic.data<float>(),
                       src2.data<float>(),
                       size,
                       max_ratio,
                       max_diff);
      return;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      data_diff_kernel(src1_basic.data<float16_t>(),
                       src2.data<float16_t>(),
                       size,
                       max_ratio,
                       max_diff);
#endif
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
      LOG(FATAL) << "data type: " << PrecisionRepr(src1_basic.precision())
                 << " is unsupported now";
  }
}

template <typename dtype>
void tensor_diff_kernel(const dtype* src1,
                        const dtype* src2,
                        dtype* dst,
                        int64_t size,
                        PrecisionType precision) {
  switch (precision) {
    case PRECISION(kFloat):
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
#endif
    case PRECISION(kInt32):
      for (int i = 0; i < size; ++i) {
        // VLOG(4) << i << "   " << src1[i] << "  " << src2[i];
        dst[i] = src1[i] - src2[i];
      }
      return;
    case PRECISION(kInt8):
      for (int i = 0; i < size; ++i) {
        dst[i] = src1[i] - src2[i];
        if (static_cast<int>(abs(dst[i])) > 0.1) {
          VLOG(4) << i << "   " << static_cast<int>(src1[i]) << "  "
                  << static_cast<int>(src2[i]);
        }
      }
      return;
    default:
      LOG(FATAL) << "data type error";
  }
}
void tensor_diff(const Tensor& t1, const Tensor& t2, Tensor& tdiff) {  // NOLINT
  int64_t size1 = t1.numel();
  int64_t size2 = t2.numel();
  int64_t size_out = tdiff.numel();
  CHECK_EQ(size1, size2) << "ERROR: tensor_diff: wrong shape";
  CHECK_EQ(size1, size_out) << "ERROR: tensor_diff: wrong shape";
  auto ptype1 = PrecisionRepr(t1.precision());
  auto ptype2 = PrecisionRepr(t2.precision());
  auto ptype3 = PrecisionRepr(tdiff.precision());
  CHECK_EQ(ptype1, ptype2) << "ERROR: tensor_diff: wrong data type";
  CHECK_EQ(ptype1, ptype3) << "ERROR: tensor_diff: wrong data type";
  switch (t1.precision()) {
    case PRECISION(kFloat):
      tensor_diff_kernel(t1.data<float>(),
                         t2.data<float>(),
                         tdiff.mutable_data<float>(),
                         size1,
                         t1.precision());
      return;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      tensor_diff_kernel(t1.data<float16_t>(),
                         t2.data<float16_t>(),
                         tdiff.mutable_data<float16_t>(),
                         size1,
                         t1.precision());
#endif
      return;
    case PRECISION(kInt32):
      tensor_diff_kernel(t1.data<int>(),
                         t2.data<int>(),
                         tdiff.mutable_data<int>(),
                         size1,
                         t1.precision());
    case PRECISION(kInt8):
      tensor_diff_kernel(t1.data<int8_t>(),
                         t2.data<int8_t>(),
                         tdiff.mutable_data<int8_t>(),
                         size1,
                         t1.precision());
      return;
    default:
      LOG(FATAL) << "data type: " << ptype1 << " is unsupported now";
  }
}

}  // namespace lite
}  // namespace paddle
