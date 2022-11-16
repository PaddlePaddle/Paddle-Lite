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

#pragma once
#include <algorithm>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/host/elementwise_op_func.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

// check whether the tensor with dimension of second can assign to the
// tensor with dimension of first
inline void CheckIsDimsMatch(const DDim& first, const DDim& second) {
  int ignore_axis1 = 0, ignore_axis2 = 0;
  for (; ignore_axis1 < first.size(); ++ignore_axis1) {
    if (first[ignore_axis1] != 1) {
      break;
    }
  }
  for (; ignore_axis2 < second.size(); ++ignore_axis2) {
    if (second[ignore_axis2] != 1) {
      break;
    }
  }

  if (second.size() == ignore_axis2) {
    // second tensor has only one value
    return;
  }

  if (first.size() - ignore_axis1 >= second.size() - ignore_axis2) {
    int idx1 = first.size() - 1;
    int idx2 = second.size() - 1;
    bool is_match = true;
    for (; idx2 >= ignore_axis2; idx2--) {
      if (first[idx1--] != second[idx2] && second[idx2] != 1) {
        is_match = false;
        break;
      }
    }
    if (is_match) {
      return;
    }
  }
  LOG(FATAL) << "The shape of tensor assigned value must match the shape of "
                "target shape: "
             << second << "but now shape is " << first << ".";
}

template <typename T = int64_t>
void CheckAndUpdateSliceAttrs(const DDim in_dims,
                              const std::vector<T>& axes,
                              std::vector<T>* starts,
                              std::vector<T>* ends,
                              std::vector<int64_t>* steps = nullptr,
                              std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    CHECK_LT(axis, in_dims.size()) << "The axis value should be less than "
                                      "the rank of input, but received axes["
                                   << i << "] = " << axis << "rank of input is "
                                   << in_dims.size() << ".";

    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      continue;
    }

    T dim_value = in_dims[axis];

    if (dim_value > 0) {
      T step = steps == nullptr ? 1 : (*steps)[i];
      CHECK_NE(step, 0) << "Step should not be 0, but received step = " << step
                        << ".";
      T start = (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = std::max(start, static_cast<T>(0));

      T end =
          0 < step && (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = std::min(end, dim_value);

      if (step > 0) {
        start = std::min(start, dim_value);
        end = std::max(end, static_cast<T>(0));
        CHECK_GE(end, start)
            << "When step > 0, end should be greater than start, but "
               "received end = "
            << end << ", start = " << start << ".";
      } else {
        start = std::min(start, dim_value - 1);
        if (end < -1) {
          end += dim_value;
        }
        end = std::max(end, static_cast<T>(-1));
        CHECK_GE(start, end)
            << "When step < 0, start should be greater than end, but "
               "received end = "
            << end << ", start = " << start << ".";
      }

      (*starts)[i] = start;
      (*ends)[i] = end;
    } else if (dim_value == 0) {
      (*starts)[i] = 0;
      (*ends)[i] = 0;
    }
  }
}

template <typename T = int64_t>
DDim GetSliceDims(const DDim in_dims,
                  const std::vector<T>& axes,
                  const std::vector<T>& starts,
                  const std::vector<T>& ends,
                  std::vector<T>* steps = nullptr,
                  std::vector<T>* infer_flags = nullptr) {
  DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      slice_dims[axis] = -1;
      continue;
    }

    if (in_dims[axis] == -1) {
      continue;
    }

    T start = starts[i];
    T end = ends[i];
    T step = steps == nullptr ? 1 : (*steps)[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

template <typename T = int64_t>
inline DDim GetDecreasedDims(const DDim slice_dims,
                             const std::vector<T>& decrease_axes,
                             std::vector<T>* infer_flags = nullptr) {
  DDim decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        CHECK_EQ(decreased_dims[axis], 1)
            << "Decrease dim should be 1, but now received "
            << decreased_dims[axis] << ".";
      }
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }

    if (new_shape.size() == 0) {
      new_shape.push_back(1);
    }

    decreased_dims = DDim(new_shape);
  }
  return decreased_dims;
}

static inline std::vector<int64_t> GetDataFromTensorList(
    const std::vector<const lite::Tensor*>& tensor_list) {
  // get tensor
  std::vector<int64_t> vec_new_data;
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    auto tensor = tensor_list[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    vec_new_data.push_back(static_cast<int64_t>(*tensor->data<int>()));
  }
  return vec_new_data;
}

template <typename T>
void stride_slice_reverse_assign(T* input,
                                 T* val,
                                 std::vector<int64_t> in_dims,
                                 std::vector<int64_t> val_dims,
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
  int val_num = val_dims[in_dims_size - 1];
  for (int i = in_dims_size - 2; i >= 0; i--) {
    dst_step[i] = val_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    val_num *= val_dims[i];
  }

  for (int dst_id = 0; dst_id < val_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < val_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id * strides_indices[j] + starts_indices[j]) * src_step[j];
    }
    input[src_id] = val[dst_id];
  }
}

template <typename D>
class SetValueCompute : public KernelLite<TARGET(kHost), PRECISION(kAny)> {
 public:
  using param_t = operators::SetValueParam;

  template <typename T, size_t RANK>
  void SetValueImpl(const lite::Tensor* input,
                    const lite::Tensor* value,
                    std::vector<int64_t>& starts,
                    std::vector<int64_t>& ends,
                    std::vector<int64_t>& steps,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& decrease_axes,
                    const std::vector<int64_t>& none_axes,
                    lite::Tensor* out) {
    auto in_dims = input->dims();
    CheckAndUpdateSliceAttrs<int64_t>(in_dims, axes, &starts, &ends, &steps);
    auto slice_dims =
        GetSliceDims<int64_t>(in_dims, axes, starts, ends, &steps);
    if (!slice_dims.production()) {
      out->CopyDataFrom(*input);
      return;
    }
    auto decrease_slice_dims =
        GetDecreasedDims<int64_t>(slice_dims, decrease_axes);
    auto slice_dims_for_assign = decrease_slice_dims;

    if (!none_axes.empty()) {
      std::vector<int64_t> slice_dims_with_none;

      size_t none_axes_cur = 0, decrease_axes_cur = 0;
      for (int i = 0; i < slice_dims.size(); ++i) {
        while (none_axes_cur < none_axes.size() &&
               none_axes[none_axes_cur] <= i) {
          slice_dims_with_none.push_back(1);
          none_axes_cur++;
        }
        if (decrease_axes_cur < decrease_axes.size() &&
            decrease_axes[decrease_axes_cur] == i) {
          decrease_axes_cur++;
        } else {
          slice_dims_with_none.push_back(slice_dims[i]);
        }
      }
      while (none_axes_cur < none_axes.size()) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      slice_dims_for_assign = DDim(slice_dims_with_none);
    }

    out->Resize(in_dims);
    out->CopyDataFrom(*input);

    lite::Tensor slice_tensor;
    lite::Tensor pad_tensor;
    slice_tensor.Resize(slice_dims);
    slice_tensor.mutable_data<T>();
    pad_tensor.Resize(in_dims);
    pad_tensor.mutable_data<T>();

    std::function<T(T, T)> SubFunc = naive_sub<T>;

    // Step 1: Set the value of out at `_index` to zero
    std::vector<int64_t> starts_indices;
    std::vector<int64_t> ends_indices;
    std::vector<int64_t> strides_indices;
    for (size_t axis = 0; axis < in_dims.size(); axis++) {
      starts_indices.push_back(0);
      ends_indices.push_back(slice_dims[axis]);
      strides_indices.push_back(1);
    }
    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices[axis_index] = starts[axis];
      ends_indices[axis_index] = ends[axis];
      strides_indices[axis_index] = steps[axis];
    }
    memset(slice_tensor.mutable_data<T>(), 0, slice_tensor.memory_size());
    stride_slice_reverse_assign(out->mutable_data<T>(),
                                slice_tensor.mutable_data<T>(),
                                in_dims.data(),
                                slice_dims.data(),
                                starts_indices,
                                ends_indices,
                                strides_indices);

    // Step 2: Set a tensor with the same shape as out tensor. And its data at
    // '_index' is the same as value, and data out of '_index' to zero
    slice_tensor.Resize(slice_dims_for_assign);
    CheckIsDimsMatch(slice_dims_for_assign, value->dims());
    // ElementwiseComputeEx can do broadcasting
    auto batch_arg0 = lite::kernels::host::GenBatchElementWiseArg<T>(
        &slice_tensor, value, &slice_tensor);
    common_elmentwise_op_naive_cpu(batch_arg0, SubFunc);
    slice_tensor.Resize(slice_dims);
    // - Step 2.2 Pad slice tensor with 0
    memset(pad_tensor.mutable_data<T>(), 0, pad_tensor.memory_size());

    stride_slice_reverse_assign(pad_tensor.mutable_data<T>(),
                                slice_tensor.mutable_data<T>(),
                                in_dims.data(),
                                slice_dims.data(),
                                starts_indices,
                                ends_indices,
                                strides_indices);

    // Step 3: Set out tensor with value
    auto batch_arg1 =
        lite::kernels::host::GenBatchElementWiseArg<T>(out, &pad_tensor, out);
    common_elmentwise_op_naive_cpu(batch_arg1, SubFunc);
  }

  template <typename T>
  void SetValue(const lite::Tensor* input,
                std::vector<int64_t>& starts,
                std::vector<int64_t>& ends,
                std::vector<int64_t>& steps,
                const std::vector<int64_t>& axes,
                const std::vector<int64_t>& decrease_axes,
                const std::vector<int64_t>& none_axes,
                const std::vector<int64_t>& shape,
                const std::vector<T>& values,
                lite::Tensor* out) {
    lite::Tensor value_tensor;
    value_tensor.Resize(shape);
    T* value_tensor_data = value_tensor.mutable_data<T>();
    std::memcpy(static_cast<void*>(value_tensor_data),
                static_cast<const void*>(values.data()),
                sizeof(T) * values.size());
    SetTensorValueKernel<T>(input,
                            &value_tensor,
                            starts,
                            ends,
                            steps,
                            axes,
                            decrease_axes,
                            none_axes,
                            out);
  }

  template <typename T>
  void SetTensorValueKernel(const lite::Tensor* input,
                            const lite::Tensor* value,
                            std::vector<int64_t>& starts,
                            std::vector<int64_t>& ends,
                            std::vector<int64_t>& steps,
                            const std::vector<int64_t>& axes,
                            const std::vector<int64_t>& decrease_axes,
                            const std::vector<int64_t>& none_axes,
                            lite::Tensor* out) {
    const int rank = input->dims().size();
    switch (rank) {
#define SET_VALUE_IMPL(rank)             \
  case rank: {                           \
    SetValueImpl<T, rank>(input,         \
                          value,         \
                          starts,        \
                          ends,          \
                          steps,         \
                          axes,          \
                          decrease_axes, \
                          none_axes,     \
                          out);          \
    break;                               \
  }
      SET_VALUE_IMPL(1)
      SET_VALUE_IMPL(2)
      SET_VALUE_IMPL(3)
      SET_VALUE_IMPL(4)
      SET_VALUE_IMPL(5)
      SET_VALUE_IMPL(6)
      default:
        LOG(FATAL) << "The rank of input should be less than 7, but received "
                   << rank;
        break;
    }
  }

  void Run() override;

  virtual ~SetValueCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
