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
#include "lite/core/tensor.h"
#include "lite/fluid/float16.h"
#include "lite/utils/paddle_enforce.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace lite {
namespace fluid {

// EigenDim converts paddle::platform::DDim into Eigen::DSizes.
template <int D>
struct EigenDim {
  using Type = Eigen::DSizes<Eigen::DenseIndex, D>;

  static Type From(const lite::DDim& dims) {
    PADDLE_ENFORCE_EQ(dims.size(), D, "D must match DDim::size");
    Type ret;
    for (size_t d = 0; d < dims.size(); d++) {
      ret[d] = dims[d];
    }
    return ret;
  }

  static Type From(const DDim::value_type length) {
    PADDLE_ENFORCE_EQ(D, 1, "D must be 1.");
    Type ret;
    ret[0] = length;
    return ret;
  }
};

// Interpret paddle::platform::Tensor as EigenTensor and EigenConstTensor.
template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenTensor {
  // TODO(qijun) Now, default type in unaligned, and we will make a benchmark on
  // the speed of aligned and unaligned version in future.
  using Type = Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, IndexType>>;

  using ConstType =
      Eigen::TensorMap<Eigen::Tensor<const T, D, MajorType, IndexType>>;

  static Type From(Tensor& tensor, const lite::DDim& dims) {  // NOLINT
    return Type(const_cast<T*>(tensor.data<T>()),
                EigenDim<D>::From(dims));  // NOLINT
  }

  static Type From(Tensor& tensor) {  // NOLINT
    return From(tensor, tensor.dims());
  }  // NOLINT

  static ConstType From(const Tensor& tensor, const lite::DDim& dims) {
    return ConstType(tensor.data<T>(), EigenDim<D>::From(dims));
  }

  static ConstType From(const Tensor& tensor) {
    return From(tensor, tensor.dims());
  }
};

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenMatrix : public EigenTensor<T, 2, MajorType, IndexType> {
  static typename EigenMatrix::Type Reshape(Tensor& tensor,  // NOLINT
                                            int num_col_dims) {
    int rank = tensor.dims().size();
    PADDLE_ENFORCE(num_col_dims > 0 && num_col_dims < rank,
                   "`num_col_dims` must be between (0, rank_of_tensor).");
    return EigenMatrix::From(tensor, tensor.dims().Flatten2D(num_col_dims));
  }

  static typename EigenMatrix::ConstType Reshape(const Tensor& tensor,
                                                 int num_col_dims) {
    int rank = tensor.dims().size();
    PADDLE_ENFORCE(num_col_dims > 0 && num_col_dims < rank,
                   "`num_col_dims` must be between (0, rank_of_tensor).");
    return EigenMatrix::From(tensor, tensor.dims().Flatten2D(num_col_dims));
  }
};

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenVector : public EigenTensor<T, 1, MajorType, IndexType> {
  // Flatten reshapes a Tensor into an EigenVector.
  static typename EigenVector::Type Flatten(Tensor& tensor) {  // NOLINT
    return typename EigenVector::Type(
        const_cast<T*>(tensor.data<T>()),
        EigenDim<1>::From(tensor.dims().production()));
  }

  static typename EigenVector::ConstType Flatten(
      const Tensor& tensor) {  // NOLINT
    return typename EigenVector::ConstType(
        tensor.data<T>(), EigenDim<1>::From(tensor.dims().production()));
  }
};

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenScalar {
  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  using Type = Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, MajorType, IndexType>>;
  using ConstType = Eigen::TensorMap<
      Eigen::TensorFixedSize<const T, Eigen::Sizes<>, MajorType, IndexType>>;

  static Type From(Tensor* tensor) {
    return Type(const_cast<T*>(tensor->data<T>()));
  }  // NOLINT

  static ConstType From(const Tensor& tensor) {
    return ConstType(tensor.data<T>());
  }
};

template <lite::TargetType Target>
struct EigenDevice;

template <>
struct EigenDevice<lite::TargetType::kX86> {
  using Type = ::Eigen::DefaultDevice;
};

template <lite::TargetType Target>
using EigenDeviceType = typename EigenDevice<Target>::Type;

}  // namespace fluid
}  // namespace lite
}  // namespace paddle
