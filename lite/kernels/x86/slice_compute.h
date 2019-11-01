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

#include <Eigen/Core>
#include <algorithm>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/fluid/eigen.h"
#include "lite/operators/relu_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <size_t D>
void slice_compute(const lite::Tensor* in,
                   lite::Tensor* out,
                   std::vector<int> axes,
                   std::vector<int> starts,
                   std::vector<int> ends,
                   std::vector<int> decrease_axis) {
  auto out_dims = out->dims();
  auto in_dims = in->dims();

  // resize out_dims
  if (decrease_axis.size() > 0) {
    if (decrease_axis.size() == (size_t)in_dims.size()) {
      std::vector<int64_t> vec_origin_out_shape(decrease_axis.size(), 1);
      // lite::DDim dims(vec_origin_out_shape);
      out->Resize(vec_origin_out_shape);
    } else {
      std::vector<int64_t> vec_origin_out_shape(
          out_dims.size() + decrease_axis.size(), -1);
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        vec_origin_out_shape[decrease_axis[i]] = 1;
      }
      int index = 0;
      for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
        if (-1 == vec_origin_out_shape[i]) {
          vec_origin_out_shape[i] = out_dims[index];
          ++index;
        }
      }
      // lite::DDim dims(vec_origin_out_shape);
      out->Resize(vec_origin_out_shape);
    }
  }

  out->mutable_data<float>(lite::TargetType::kX86);

  auto new_out_dims = out->dims();
  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }
  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      lite::fluid::EigenTensor<float, D, Eigen::RowMajor, Eigen::DenseIndex>::
          From(*in, in->dims());
  auto out_t =
      lite::fluid::EigenTensor<float, D, Eigen::RowMajor, Eigen::DenseIndex>::
          From(*out, new_out_dims);
  out_t = in_t.slice(offsets, extents);

  out->Resize(out_dims);
}

template <typename T>
void slice_compute_(const lite::Tensor* Input,
                    lite::Tensor* Out,
                    std::vector<int> axes,
                    std::vector<int> starts,
                    std::vector<int> ends,
                    std::vector<int> decrease_axis) {
  int rank = Input->dims().size();
  switch (rank) {
    case 1:
      slice_compute<1>(Input, Out, axes, starts, ends, decrease_axis);
      break;
    case 2:
      slice_compute<2>(Input, Out, axes, starts, ends, decrease_axis);
      break;
    case 3:
      slice_compute<3>(Input, Out, axes, starts, ends, decrease_axis);
      break;
    case 4:
      slice_compute<4>(Input, Out, axes, starts, ends, decrease_axis);
      break;
    case 5:
      slice_compute<5>(Input, Out, axes, starts, ends, decrease_axis);
      break;
    case 6:
      slice_compute<6>(Input, Out, axes, starts, ends, decrease_axis);
      break;
  }
}

template <typename T>
class SliceCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SliceParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    slice_compute_<T>(param.X,
                      param.Out,
                      param.axes,
                      param.starts,
                      param.ends,
                      param.decrease_axis);
  }

  virtual ~SliceCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
