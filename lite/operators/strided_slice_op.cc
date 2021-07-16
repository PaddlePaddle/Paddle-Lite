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

#include "lite/operators/strided_slice_op.h"
#include <algorithm>
#include <cstddef>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool StridedSliceOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  CHECK_OR_FALSE(param_.Out);
  auto in_dims = param_.Input->dims();
  CHECK_LT(in_dims.size(), 7) << "input_dims.size(): " << in_dims.size()
                              << " should be less than 7.";
  return true;
}
static std::vector<int64_t> StridedSliceOutDims(
    const std::vector<int> starts,
    const std::vector<int> ends,
    const std::vector<int> strides,
    const std::vector<int> axes,
    const std::vector<int> infer_flags,
    const DDim in_dims,
    const std::vector<int> decrease_axis,
    const size_t size,
    bool infer_shape) {
  std::vector<int64_t> out_dims_vector;
  for (int i = 0; i < in_dims.size(); i++) {
    out_dims_vector.push_back(in_dims[i]);
  }
  int stride_index;
  int start_index;
  int end_index;
  for (size_t i = 0; i < size; i++) {
    int axes_index = axes[i];
    start_index = starts[i];
    end_index = ends[i];
    stride_index = strides[i];
    bool decrease_axis_affect = false;
    if (start_index == -1 && end_index == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    if (decrease_axis_affect) {
      out_dims_vector[axes_index] = 1;
      continue;
    }
    if (infer_shape && infer_flags[i] == -1) {
      out_dims_vector[axes_index] = -1;
      continue;
    }
    CHECK_NE(stride_index, 0) << "stride index in StridedSlice operator is 0.";
    CHECK_LT(axes_index, in_dims.size())
        << "axes_index: " << axes_index
        << " should be less than in_dims.size(): " << in_dims.size() << ".";
    int64_t axis_size = in_dims[axes_index];

    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {  // skip None stop condition
        end_index = end_index + axis_size;
      }
    }

    if (stride_index < 0) {
      start_index = start_index + 1;
      end_index = end_index + 1;
    }

    bool zero_dim_condition =
        ((stride_index < 0 && (start_index <= end_index)) ||
         (stride_index > 0 && (start_index >= end_index)));
    CHECK_EQ(zero_dim_condition, false) << "The start index and end index are "
                                           "invalid for their corresponding "
                                           "stride.";
    auto tmp = std::max(start_index, end_index);
    int32_t left =
        std::max(static_cast<int32_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, static_cast<int64_t>(tmp));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
  return out_dims_vector;
}

bool StridedSliceOp::InferShapeImpl() const {
  auto input = param_.Input;
  auto input_dims = input->dims();
  auto starts = param_.starts;
  auto ends = param_.ends;
  auto strides = param_.strides;
  auto axes = param_.axes;
  auto infer_flags = param_.infer_flags;
  auto decrease_axis = param_.decrease_axis;
  auto starts_size = starts.size();
  auto ends_size = ends.size();
  auto strides_size = strides.size();

  if (param_.StartsTensorList.size() > 0) {
    starts_size = param_.StartsTensorList.size();
  }
  if (param_.EndsTensorList.size() > 0) {
    ends_size = param_.EndsTensorList.size();
  }
  if (param_.StridesTensorList.size() > 0) {
    strides_size = param_.StridesTensorList.size();
  }
  std::vector<int64_t> out_dims_vector(input_dims.size(), -1);
  if (!param_.tensor_input) {
    out_dims_vector = StridedSliceOutDims(starts,
                                          ends,
                                          strides,
                                          axes,
                                          infer_flags,
                                          input_dims,
                                          decrease_axis,
                                          axes.size(),
                                          true);
  }
  auto out_dims = DDim(out_dims_vector);
  // generate new shape
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      if (infer_flags[i] != -1) {
        CHECK_EQ(out_dims[decrease_axis[i]], 1)
            << "the size of decrease dimension should be 1, "
            << "but received " << out_dims[decrease_axis[i]] << ".";
      }
      out_dims[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims.size(); ++i) {
      if (out_dims[i] != 0) {
        new_out_shape.push_back(out_dims[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }

    out_dims = DDim(new_out_shape);
  }
  param_.Out->Resize(out_dims);
  return true;
}

bool StridedSliceOp::AttachImpl(const cpp::OpDesc &op_desc,
                                lite::Scope *scope) {
  param_.Input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  if (op_desc.HasAttr("starts")) {
    param_.starts = op_desc.GetAttr<std::vector<int>>("starts");
  }

  if (op_desc.HasAttr("ends")) {
    param_.ends = op_desc.GetAttr<std::vector<int>>("ends");
  }

  if (op_desc.HasAttr("strides")) {
    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  }

  if (op_desc.HasAttr("axes")) {
    param_.axes = op_desc.GetAttr<std::vector<int>>("axes");
  }

  if (op_desc.HasAttr("infer_flags")) {
    param_.infer_flags = op_desc.GetAttr<std::vector<int>>("infer_flags");
  }
  if (op_desc.HasAttr("decrease_axis")) {
    param_.decrease_axis = op_desc.GetAttr<std::vector<int>>("decrease_axis");
  }

  auto starts_size = param_.starts.size();
  auto ends_size = param_.ends.size();
  auto strides_size = param_.strides.size();
  if (op_desc.HasInput("StartsTensorList") &&
      !op_desc.Input("StartsTensorList").empty()) {
    param_.StartsTensorList.clear();
    auto inputs = op_desc.Input("StartsTensorList");
    for (auto var : inputs) {
      param_.StartsTensorList.push_back(
          scope->FindVar(var)->GetMutable<lite::Tensor>());
    }
  }
  if (op_desc.HasInput("EndsTensorList") &&
      !op_desc.Input("EndsTensorList").empty()) {
    param_.EndsTensorList.clear();
    auto inputs = op_desc.Input("EndsTensorList");
    for (auto var : inputs) {
      param_.EndsTensorList.push_back(
          scope->FindVar(var)->GetMutable<lite::Tensor>());
    }
  }
  if (op_desc.HasInput("StridesTensorList") &&
      !op_desc.Input("StridesTensorList").empty()) {
    param_.StridesTensorList.clear();
    auto inputs = op_desc.Input("StridesTensorList");
    for (auto var : inputs) {
      param_.StridesTensorList.push_back(
          scope->FindVar(var)->GetMutable<lite::Tensor>());
    }
  }
  auto tensor_input = false;
  if ((op_desc.HasInput("EndsTensor") &&
       !op_desc.Input("EndsTensor").empty()) ||
      (op_desc.HasInput("StartsTensor") &&
       !op_desc.Input("StartsTensor").empty()) ||
      (op_desc.HasInput("StridesTensor") &&
       !op_desc.Input("StridesTensor").empty())) {
    tensor_input = true;
  }
  param_.tensor_input = tensor_input;
  if (op_desc.HasInput("EndsTensor") && !op_desc.Input("EndsTensor").empty()) {
    auto inputs = op_desc.Input("EndsTensor").front();
    param_.EndsTensor = scope->FindVar(inputs)->GetMutable<Tensor>();
  } else {
    CHECK_EQ(param_.axes.size(), ends_size)
        << "axes.size(): " << param_.axes.size()
        << " is not equal to ends_size: " << ends_size;
  }
  if (op_desc.HasInput("StartsTensor") &&
      !op_desc.Input("StartsTensor").empty()) {
    auto inputs = op_desc.Input("StartsTensor").front();
    param_.StartsTensor = scope->FindVar(inputs)->GetMutable<Tensor>();
  } else {
    CHECK_EQ(param_.axes.size(), starts_size)
        << "axes.size(): " << param_.axes.size()
        << " is not equal to starts_size: " << starts_size;
  }
  if (op_desc.HasInput("StridesTensor") &&
      !op_desc.Input("StridesTensor").empty()) {
    auto inputs = op_desc.Input("StridesTensor").front();
    param_.StridesTensor = scope->FindVar(inputs)->GetMutable<Tensor>();
  } else {
    CHECK_EQ(param_.axes.size(), strides_size)
        << "axes.size(): " << param_.axes.size()
        << " is not equal to ends_size: " << strides_size;
  }
  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(strided_slice, paddle::lite::operators::StridedSliceOp);
