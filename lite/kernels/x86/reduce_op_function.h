// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/fluid/eigen.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = lite::fluid::EigenTensor<T, D, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = lite::fluid::EigenScalar<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = lite::fluid::EigenVector<T, MajorType, IndexType>;

template <lite::TargetType Target,
          typename T,
          size_t D,
          size_t R_D,
          typename Functor>
// const lite::Context<Target>& context,
void ReduceFunctor(const lite::Tensor& input,
                   lite::Tensor* output,
                   const std::vector<int>& dims,
                   bool keep_dim) {
  auto x = EigenTensor<T, D>::From(input);

  auto reduce_dim = Eigen::array<int, R_D>();
  auto x_rank = static_cast<int>(x.dimensions().size());
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      reduce_dim[i] = x_rank + dims[i];
    } else {
      reduce_dim[i] = dims[i];
    }
  }

  Functor functor;
  if (D == 1) {
    auto out = EigenScalar<T>::From(output);
    functor(&x, &out, reduce_dim);
  } else {
    std::vector<DDim::value_type> out_dims;
    if (keep_dim) {
      // Construct the squeezed dims.
      const int kDelFlag = -2;
      out_dims = output->dims().Vectorize();
      for (size_t i = 0; i < dims.size(); ++i) {
        out_dims[reduce_dim[i]] = kDelFlag;
      }
      out_dims.erase(remove(out_dims.begin(), out_dims.end(), kDelFlag),
                     out_dims.end());
    }
    auto out = EigenTensor<T, (D - R_D)>::From(
        *output, keep_dim ? DDim(out_dims) : output->dims());
    functor(&x, &out, reduce_dim);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
