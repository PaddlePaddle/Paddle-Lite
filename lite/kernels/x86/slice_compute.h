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

inline std::vector<int> get_new_data_from_tensorlist(
    const std::vector<lite::Tensor*>& list_new_data_tensor) {
  // get tensor from
  std::vector<int> vec_new_data;
  for (size_t i = 0; i < list_new_data_tensor.size(); ++i) {
    auto tensor = list_new_data_tensor[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    vec_new_data.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }
  return vec_new_data;
}

inline std::vector<int> get_new_data_from_tensor(
    const Tensor* new_data_tensor) {
  std::vector<int> vec_new_data;
  auto* new_data = new_data_tensor->data<int>();
  vec_new_data =
      std::vector<int>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

template <size_t D>
void slice_compute(const lite::Tensor* in,
                   lite::Tensor* out,
                   std::vector<int> axes,
                   std::vector<int> starts,
                   std::vector<int> ends,
                   std::vector<int> decrease_axis,
                   lite::Tensor* StartsTensor,
                   lite::Tensor* EndsTensor,
                   std::vector<lite::Tensor*> StartsTensorList,
                   std::vector<lite::Tensor*> EndsTensorList,
                   std::vector<int> infer_flags) {
  auto out_dims = out->dims();
  auto in_dims = in->dims();

  bool need_infer = false;
  if (StartsTensor || EndsTensor) {
    need_infer = true;
  } else if (StartsTensorList.size() > 0 || EndsTensorList.size() > 0) {
    need_infer = true;
  }

  if (need_infer) {
    if (StartsTensor) {
      starts = get_new_data_from_tensor(StartsTensor);
    } else if (StartsTensorList.size() > 0) {
      starts = get_new_data_from_tensorlist(StartsTensorList);
    }
    CHECK_EQ(starts.size(), axes.size())
        << "The size of starts must be equal to the size of axes.";
    if (EndsTensor) {
      ends = get_new_data_from_tensor(EndsTensor);
    } else if (EndsTensorList.size() > 0) {
      ends = get_new_data_from_tensorlist(EndsTensorList);
    }
    CHECK_EQ(ends.size(), axes.size())
        << "The size of ends must be equal to the size of axes.";
    out_dims = in_dims;
    int dim_value, start, end;
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = out_dims[axes[i]];
      if (dim_value > 0) {
        // when end = start + 1 and start == -1
        if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
          auto ret =
              std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
          if (ret != decrease_axis.end()) {
            ends[i] = 10000000;
          }
        }

        start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
        end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
        start = (std::max)(start, 0);
        end = (std::max)(end, 0);
        end = (std::min)(end, dim_value);
        CHECK_GT(end, start) << "end should greater than start";
        out_dims[axes[i]] = end - start;
      }
    }
    out->Resize(out_dims);
    // generate new shape
    if (decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        CHECK_EQ(out_dims[decrease_axis[i]], 1) << "decrease dim should be 1";
        out_dims[decrease_axis[i]] = 0;
      }

      for (size_t i = 0; i < out_dims.size(); ++i) {
        if (out_dims[i] != 0) {
          new_out_shape.push_back(out_dims[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }

      DDim new_dims;
      new_dims.ConstructFrom(new_out_shape);
      out_dims = new_dims;
    }
  }

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

  out->mutable_data<float>();

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
    start = (std::max)(start, 0);
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
                    std::vector<int> decrease_axis,
                    lite::Tensor* StartsTensor,
                    lite::Tensor* EndsTensor,
                    std::vector<lite::Tensor*> StartsTensorList,
                    std::vector<lite::Tensor*> EndsTensorList,
                    std::vector<int> infer_flags) {
  int rank = Input->dims().size();
  switch (rank) {
    case 1:
      slice_compute<1>(Input,
                       Out,
                       axes,
                       starts,
                       ends,
                       decrease_axis,
                       StartsTensor,
                       EndsTensor,
                       StartsTensorList,
                       EndsTensorList,
                       infer_flags);
      break;
    case 2:
      slice_compute<2>(Input,
                       Out,
                       axes,
                       starts,
                       ends,
                       decrease_axis,
                       StartsTensor,
                       EndsTensor,
                       StartsTensorList,
                       EndsTensorList,
                       infer_flags);
      break;
    case 3:
      slice_compute<3>(Input,
                       Out,
                       axes,
                       starts,
                       ends,
                       decrease_axis,
                       StartsTensor,
                       EndsTensor,
                       StartsTensorList,
                       EndsTensorList,
                       infer_flags);
      break;
    case 4:
      slice_compute<4>(Input,
                       Out,
                       axes,
                       starts,
                       ends,
                       decrease_axis,
                       StartsTensor,
                       EndsTensor,
                       StartsTensorList,
                       EndsTensorList,
                       infer_flags);
      break;
    case 5:
      slice_compute<5>(Input,
                       Out,
                       axes,
                       starts,
                       ends,
                       decrease_axis,
                       StartsTensor,
                       EndsTensor,
                       StartsTensorList,
                       EndsTensorList,
                       infer_flags);
      break;
    case 6:
      slice_compute<6>(Input,
                       Out,
                       axes,
                       starts,
                       ends,
                       decrease_axis,
                       StartsTensor,
                       EndsTensor,
                       StartsTensorList,
                       EndsTensorList,
                       infer_flags);
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
                      param.decrease_axis,
                      param.StartsTensor,
                      param.EndsTensor,
                      param.StartsTensorList,
                      param.EndsTensorList,
                      param.infer_flags);
  }

  virtual ~SliceCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
