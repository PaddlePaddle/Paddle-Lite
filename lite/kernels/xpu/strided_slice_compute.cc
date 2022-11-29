// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/xpu/strided_slice_compute.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

// get the output Dims
inline std::vector<int64_t> StridedSliceOutDims(
    const std::vector<int> starts,
    const std::vector<int> ends,
    const std::vector<int> strides,
    const std::vector<int> axes,
    const std::vector<int> infer_flags,
    const std::vector<int64_t> in_dims,
    const std::vector<int> decrease_axis,
    const size_t size,
    bool infer_shape) {
  std::vector<int64_t> out_dims_vector;
  for (size_t i = 0; i < in_dims.size(); i++) {
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
      if (!(end_index == -1 && stride_index < 0)) {
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
    int64_t step = std::abs(static_cast<int64_t>(stride_index));
    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
  return out_dims_vector;
}

inline std::vector<int> StridedSliceFunctor(
    int* starts,
    int* ends,
    int* strides,
    int* axes,
    std::vector<int64_t> dims,
    const std::vector<int> infer_flags,
    const std::vector<int> decrease_axis,
    const size_t size) {
  std::vector<int> reverse_axis;
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
    if (ends[axis] > dims[axes[axis]]) {
      ends[axis] = dims[axes[axis]];
    }
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }

    bool decrease_axis_affect = false;
    if (starts[axis_index] == -1 && ends[axis_index] == 0 &&
        infer_flags[axis_index] == -1) {
      auto ret = std::find(
          decrease_axis.begin(), decrease_axis.end(), axes[axis_index]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
    }
    if (ends[axis_index] < 0) {
      if (!(ends[axis_index] == -1 &&
            strides[axis_index] < 0)) {  // skip None stop condition
        ends[axis_index] = ends[axis_index] + axis_size;
      }
    }
    if (decrease_axis_affect) {
      if (strides[axis_index] < 0) {
        ends[axis_index] = starts[axis_index] - 1;
      } else {
        ends[axis_index] = starts[axis_index] + 1;
      }
    }

    if (strides[axis_index] < 0) {
      reverse_axis.push_back(*(axes + axis_index));
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        starts[axis_index] = starts[axis_index] + 1;
        ends[axis_index] = ends[axis_index] + 1;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    }
  }
  return reverse_axis;
}

inline std::vector<int> get_new_data_from_tensorlist(
    const std::vector<lite::Tensor*>& list_new_data_tensor) {
  std::vector<int> vec_new_data;
  for (size_t i = 0; i < list_new_data_tensor.size(); ++i) {
    auto tensor = list_new_data_tensor[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    vec_new_data.push_back(static_cast<int>(*tensor->data<int>()));
  }
  return vec_new_data;
}

inline std::vector<int> get_new_data_from_tensor(
    const lite::Tensor* new_data_tensor) {
  std::vector<int> vec_new_data;
  auto* new_data = new_data_tensor->data<int>();
  vec_new_data =
      std::vector<int>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

template <typename T, PrecisionType PType>
void StridedSliceCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::StridedSliceParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto input = param.Input;
  auto input_dims = input->dims();
  auto starts = param.starts;
  auto ends = param.ends;
  auto strides = param.strides;
  auto axes = param.axes;
  auto infer_flags = param.infer_flags;
  auto decrease_axis = param.decrease_axis;

  if (param.StartsTensorList.size() > 0) {
    starts = get_new_data_from_tensorlist(param.StartsTensorList);
  } else if (param.StartsTensor) {
    starts = get_new_data_from_tensor(param.StartsTensor);
  }

  if (param.EndsTensorList.size() > 0) {
    ends = get_new_data_from_tensorlist(param.EndsTensorList);
  } else if (param.EndsTensor) {
    ends = get_new_data_from_tensor(param.EndsTensor);
  }

  if (param.StridesTensorList.size() > 0) {
    strides = get_new_data_from_tensorlist(param.StridesTensorList);
  } else if (param.StridesTensor) {
    strides = get_new_data_from_tensor(param.StridesTensor);
  }

  std::vector<int64_t> out_dims_vector(input_dims.size(), -1);
  if (!param.tensor_input) {
    out_dims_vector = StridedSliceOutDims(starts,
                                          ends,
                                          strides,
                                          axes,
                                          infer_flags,
                                          input_dims.data(),
                                          decrease_axis,
                                          axes.size(),
                                          true);
  }

  auto out_dims = DDim(out_dims_vector);

  std::vector<int> reverse_axis = StridedSliceFunctor(starts.data(),
                                                      ends.data(),
                                                      strides.data(),
                                                      axes.data(),
                                                      input_dims.data(),
                                                      infer_flags,
                                                      decrease_axis,
                                                      starts.size());

  std::vector<int> starts_indices;
  std::vector<int> ends_indices;
  std::vector<int> strides_indices;

  for (size_t axis = 0; axis < input_dims.size(); axis++) {
    starts_indices.push_back(0);
    ends_indices.push_back(out_dims[axis]);
    strides_indices.push_back(1);
  }

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts[axis];
    ends_indices[axis_index] = ends[axis];
    strides_indices[axis_index] = strides[axis];
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      CHECK_EQ(out_dims[decrease_axis[i]], 1)
          << "the size of decrease dimension should be 1, but received: "
          << out_dims[decrease_axis[i]];
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (size_t i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = DDim(new_out_shape);
  }

  bool need_reverse = false;
  if (reverse_axis.size() > 0) {
    need_reverse = true;
  }

  std::vector<int> x_shape;
  x_shape.reserve(input_dims.size());
  for (int i = 0; i < input_dims.size(); i++) {
    x_shape.push_back(input_dims[i]);
  }

  param.Out->Resize(out_dims);
  auto* in_t = input->template data<T>();
  auto* out_t = param.Out->template mutable_data<T>(TARGET(kXPU));

  if (need_reverse) {
    lite::Tensor* tmp = new lite::Tensor();
    tmp->Resize(out_dims);
    auto* tmp_t = tmp->template mutable_data<T>(TARGET(kXPU));
    int r = xdnn::strided_slice<T>(ctx.GetRawContext(),
                                   in_t,
                                   tmp_t,
                                   x_shape,
                                   starts_indices,
                                   ends_indices,
                                   strides_indices);
    CHECK_EQ(r, 0);

    std::vector<int> out_dims_int(out_dims.size(), 0);
    auto out_dims_int64 = out_dims.data();
    for (int i = 0; i < out_dims_int.size(); i++) {
      out_dims_int[i] = out_dims_int64[i];
    }

    std::vector<float> vc(4, 0);
    TargetWrapperXPU::MemcpySync(
        vc.data(), tmp_t, sizeof(float) * 4, IoDirection::DtoH);

    r = xdnn::flip<T>(
        ctx.GetRawContext(), tmp_t, out_t, out_dims_int, reverse_axis);
    CHECK_EQ(r, 0);

    TargetWrapperXPU::MemcpySync(
        vc.data(), out_t, sizeof(float) * 4, IoDirection::DtoH);
  } else {
    int r = xdnn::strided_slice<T>(ctx.GetRawContext(),
                                   in_t,
                                   out_t,
                                   x_shape,
                                   starts_indices,
                                   ends_indices,
                                   strides_indices);
    CHECK_EQ(r, 0);
  }

  if (decrease_axis.size() > 0) {
    param.Out->Resize(out_dims_origin);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using StridedSliceFloat32 =
    paddle::lite::kernels::xpu::StridedSliceCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    strided_slice, kXPU, kFloat, kNCHW, StridedSliceFloat32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})

    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

using StridedSliceFloat16 =
    paddle::lite::kernels::xpu::StridedSliceCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(
    strided_slice, kXPU, kFP16, kNCHW, StridedSliceFloat16, def_fp16)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using StridedSliceInt32 =
    paddle::lite::kernels::xpu::StridedSliceCompute<int32_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    strided_slice, kXPU, kFloat, kNCHW, StridedSliceInt32, def_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
