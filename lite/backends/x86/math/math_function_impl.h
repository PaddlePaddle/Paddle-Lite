/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "lite/backends/x86/fluid/data_type.h"
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/math/math_function.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <lite::TargetType Target, typename T>
void SetConstant<Target, T>::operator()(const lite::Context<Target>& context,
                                        lite::Tensor* tensor,
                                        T num) {
  auto t = lite::fluid::EigenVector<T>::Flatten(*tensor);

  // t.device(*Eigen::DefaultDevice()) = t.constant(static_cast<T>(num));
  // t.device(*context.eigen_device()) = t.constant(static_cast<T>(num));
  t.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      t.constant(static_cast<T>(num));
}

template <lite::TargetType Target, typename T, int Rank>
void Transpose<Target, T, Rank>::operator()(
    const lite::Context<Target>& context,
    const lite::TensorLite& in,
    lite::TensorLite* out,
    const std::vector<int>& axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
  auto eigen_in = lite::fluid::EigenTensor<T, Rank>::From(in);
  auto eigen_out = lite::fluid::EigenTensor<T, Rank>::From(*out);
  // auto* dev = context.eigen_device();
  // eigen_out.device(*dev) = eigen_in.shuffle(permute);
  eigen_out.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      eigen_in.shuffle(permute);
}

template <lite::TargetType Target, typename T>
void ColwiseSum<Target, T>::operator()(const lite::Context<Target>& context,
                                       const lite::TensorLite& input,
                                       lite::TensorLite* out) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  CHECK_EQ(out->numel(), size);

  auto in = lite::fluid::EigenMatrix<T>::From(input);
  auto vec = lite::fluid::EigenVector<T>::Flatten(*out);

  // vec.device(*context.eigen_device()) = in.sum(Eigen::array<int, 1>({{0}}));
  vec.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      in.sum(Eigen::array<int, 1>({{0}}));
}

// Specialize for CPU, since Eigen implement a general reduce. However,
// colwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class ColwiseSum<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::TensorLite& input,
                  lite::TensorLite* out) {
    auto& in_dims = input.dims();
    auto height = in_dims[0];
    auto size = in_dims[1];
    CHECK_EQ(out->numel(), size);

    T* out_buf = out->template mutable_data<T>(out->target());
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        if (i == 0) {
          out_buf[j] = in_buf[i * size + j];
        } else {
          out_buf[j] += in_buf[i * size + j];
        }
      }
    }
  }
};

template <lite::TargetType Target, typename T>
void RowwiseMean<Target, T>::operator()(const lite::Context<Target>& context,
                                        const lite::TensorLite& input,
                                        lite::TensorLite* out) {
  auto in_dims = input.dims();
  CHECK_EQ(in_dims.size(), 2U);
  CHECK_EQ(out->numel(), in_dims[0]);

  auto in = lite::fluid::EigenMatrix<T>::From(input);
  auto vec = lite::fluid::EigenVector<T>::Flatten(*out);

  // vec.device(*context.eigen_device()) = in.mean(Eigen::array<int, 1>({{1}}));
  vec.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      in.mean(Eigen::array<int, 1>({{1}}));
}
// TODO(zcd): Following ColwiseSum format, need to confirm.
// Specialize for CPU, since Eigen implement a general reduce. However,
// rowwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class RowwiseMean<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::TensorLite& input,
                  lite::TensorLite* out) {
    auto& in_dims = input.dims();
    CHECK_EQ(in_dims.size(), 2U);
    auto height = in_dims[0];
    auto size = in_dims[1];
    CHECK_EQ(out->numel(), height);
    auto inv_size = 1.0 / size;
    T* out_buf = out->template mutable_data<T>(out->target());
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      T sum = 0;
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        sum += in_buf[i * size + j];
      }
      out_buf[i] = sum * inv_size;
    }
  }
};

template <lite::TargetType Target, typename T>
void RowwiseSum<Target, T>::operator()(const lite::Context<Target>& context,
                                       const lite::TensorLite& input,
                                       lite::TensorLite* out) {
  auto in_dims = input.dims();
  CHECK_EQ(in_dims.size(), 2U);
  CHECK_EQ(out->numel(), in_dims[0]);

  auto in = lite::fluid::EigenMatrix<T>::From(input);
  auto vec = lite::fluid::EigenVector<T>::Flatten(*out);

  // vec.device(*context.eigen_device()) = in.sum(Eigen::array<int, 1>({{1}}));
  vec.device(typename lite::fluid::EigenDevice<Target>::Type()) =
      in.sum(Eigen::array<int, 1>({{1}}));
}
// TODO(zcd): Following ColwiseSum format, need to confirm.
// Specialize for CPU, since Eigen implement a general reduce. However,
// rowwise-sum can be easily implemented. General reduce has a huge overhead in
// CPU
template <typename T>
class RowwiseSum<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::TensorLite& input,
                  lite::TensorLite* out) {
    auto& in_dims = input.dims();
    CHECK_EQ(in_dims.size(), 2U);
    auto height = in_dims[0];
    auto size = in_dims[1];
    CHECK_EQ(out->numel(), height);

    T* out_buf = out->template mutable_data<T>(out->target());
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      T sum = 0;
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        sum += in_buf[i * size + j];
      }
      out_buf[i] = sum;
    }
  }
};

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
