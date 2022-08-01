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

#include "lite/kernels/host/strided_slice_compute.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

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
    int64_t step = std::abs(static_cast<int64_t>(stride_index));

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
  return out_dims_vector;
}

inline void StridedSliceFunctor(int* starts,
                                int* ends,
                                int* strides,
                                int* axes,
                                int* reverse_axis,
                                std::vector<int64_t> dims,
                                const std::vector<int> infer_flags,
                                const std::vector<int> decrease_axis,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
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
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        starts[axis_index] = starts[axis_index] + 1;
        ends[axis_index] = ends[axis_index] + 1;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

inline std::vector<int> get_new_data_from_tensorlist(
    const std::vector<lite::Tensor*>& list_new_data_tensor) {
  // get tensor
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

template <typename T>
void stride_slice(const T* input,
                  T* out,
                  std::vector<int64_t> in_dims,
                  std::vector<int64_t> out_dims,
                  std::vector<int64_t> starts_indices,
                  std::vector<int64_t> ends_indices,
                  std::vector<int64_t> strides_indices) {
  size_t in_dims_size = in_dims.size();
  std::vector<int> dst_step;
  std::vector<int> src_step;
  for (size_t i = 0; i < in_dims_size; ++i) {
    dst_step.push_back(1);
    src_step.push_back(1);
  }
  int out_num = out_dims[in_dims_size - 1];
  for (int i = in_dims_size - 2; i >= 0; i--) {
    dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    out_num *= out_dims[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < out_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id * strides_indices[j] + starts_indices[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

template <typename T>
void reverse(const T* input,
             T* out,
             std::vector<int64_t> in_dims,
             std::vector<bool> reverse_axis) {
  const T* in_ptr = input;
  T* out_ptr = out;
  size_t in_dims_size = in_dims.size();
  std::vector<int> src_step;
  for (size_t i = 0; i < in_dims_size; ++i) {
    src_step.push_back(1);
  }
  for (int i = in_dims_size - 2; i >= 0; i--) {
    src_step[i] *= in_dims[i + 1] * src_step[i + 1];
  }
  for (size_t i = 0; i < reverse_axis.size(); i++) {
    if (reverse_axis[i]) {
      // reverse
      for (int j = 0; j < in_dims[i]; j++) {
        int size = 1;
        if (i + 1 < in_dims_size) {
          size = src_step[i + 1];
        }
        const T* in_ptr1 = in_ptr + j * size;
        T* out_ptr1 = out_ptr + (in_dims[i] - 1 - j) * size;
        memcpy(out_ptr1, in_ptr1, sizeof(T) * size);
      }
    }
    in_ptr += src_step[i];
    out_ptr += src_step[i];
  }
}

template <typename T, PrecisionType PType>
void StridedSliceCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::StridedSliceParam>();
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
  std::vector<int> reverse_vector(starts.size(), 0);
  StridedSliceFunctor(starts.data(),
                      ends.data(),
                      strides.data(),
                      axes.data(),
                      reverse_vector.data(),
                      input_dims.data(),
                      infer_flags,
                      decrease_axis,
                      starts.size());

  std::vector<int64_t> starts_indices;
  std::vector<int64_t> ends_indices;
  std::vector<int64_t> strides_indices;
  std::vector<bool> reverse_axis;
  for (size_t axis = 0; axis < input_dims.size(); axis++) {
    starts_indices.push_back(0);
    ends_indices.push_back(out_dims[axis]);
    strides_indices.push_back(1);
    reverse_axis.push_back(false);
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts[axis];
    ends_indices[axis_index] = ends[axis];
    strides_indices[axis_index] = strides[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
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
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }
  param.Out->Resize(out_dims);
  auto* in_t = input->template data<T>();
  auto* out_t = param.Out->template mutable_data<T>();
  if (need_reverse) {
    lite::Tensor* tmp = new lite::Tensor();
    tmp->Resize(out_dims);
    auto* tmp_t = tmp->mutable_data<T>();
    stride_slice(in_t,
                 tmp_t,
                 input_dims.data(),
                 out_dims.data(),
                 starts_indices,
                 ends_indices,
                 strides_indices);
    reverse(tmp_t, out_t, out_dims.data(), reverse_axis);
  } else {
    stride_slice(in_t,
                 out_t,
                 input_dims.data(),
                 out_dims.data(),
                 starts_indices,
                 ends_indices,
                 strides_indices);
  }

  if (decrease_axis.size() > 0) {
    param.Out->Resize(out_dims_origin);
  }
}

} /* namespace host */
} /* namespace kernels */
} /* namespace lite */
} /* namespace paddle */

using slice_float =
    paddle::lite::kernels::host::StridedSliceCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(strided_slice, kHost, kFloat, kNCHW, slice_float, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

using slice_int32 =
    paddle::lite::kernels::host::StridedSliceCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    strided_slice, kHost, kFloat, kNCHW, slice_int32, def_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using slice_int64 =
    paddle::lite::kernels::host::StridedSliceCompute<int64_t,
                                                     PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    strided_slice, kHost, kFloat, kNCHW, slice_int64, def_int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
