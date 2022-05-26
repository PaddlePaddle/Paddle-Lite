// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

inline std::vector<int64_t> vector_int2int64_t(std::vector<int> input) {
  std::vector<int64_t> output{};
  for (auto i : input) {
    output.push_back(static_cast<int64_t>(i));
  }
  return output;
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

class StridedSliceComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "Input";
  std::string output_ = "Out";
  std::vector<int> axes_;
  std::vector<int> starts_;
  std::vector<int> ends_;
  std::vector<int> strides_;
  std::vector<int64_t> starts_i64;
  std::vector<int64_t> ends_i64;
  std::vector<int> decrease_axis_;
  DDim dims_;
  std::vector<int> infer_flags_;
  std::string starts_tensor_ = "StartsTensor";
  std::string ends_tensor_ = "EndsTensor";
  bool use_tensor_;
  bool use_tensor_list_;

 public:
  StridedSliceComputeTester(const Place& place,
                            const std::string& alias,
                            const std::vector<int>& axes,
                            const std::vector<int>& starts,
                            const std::vector<int>& ends,
                            const std::vector<int>& strides,
                            const std::vector<int>& decrease_axis,
                            const DDim& dims,
                            bool use_tensor = false,
                            bool use_tensor_list = false,
                            const std::vector<int>& infer_flags = {})
      : TestCase(place, alias),
        axes_(axes),
        starts_(starts),
        ends_(ends),
        strides_(strides),
        decrease_axis_(decrease_axis),
        dims_(dims),
        infer_flags_(infer_flags),
        use_tensor_(use_tensor),
        use_tensor_list_(use_tensor_list) {
    this->starts_i64 = vector_int2int64_t(starts);
    this->ends_i64 = vector_int2int64_t(ends);
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    auto* input = scope->FindTensor(input_);
    CHECK(out);
    CHECK(input);
    const auto* input_data = input->data<float>();
    auto in_dims = input->dims();

    std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
    out_dims_vector = StridedSliceOutDims(starts_,
                                          ends_,
                                          strides_,
                                          axes_,
                                          infer_flags_,
                                          in_dims.data(),
                                          decrease_axis_,
                                          axes_.size(),
                                          true);
    auto out_dims = DDim(out_dims_vector);
    out->Resize(out_dims);
    auto* out_data = out->mutable_data<float>();
    std::vector<int> reverse_vector(starts_.size(), 0);
    StridedSliceFunctor(starts_.data(),
                        ends_.data(),
                        strides_.data(),
                        axes_.data(),
                        reverse_vector.data(),
                        in_dims.data(),
                        infer_flags_,
                        decrease_axis_,
                        starts_.size());

    std::vector<int64_t> starts_indices;
    std::vector<int64_t> ends_indices;
    std::vector<int64_t> strides_indices;
    std::vector<bool> reverse_axis;
    for (size_t axis = 0; axis < in_dims.size(); axis++) {
      starts_indices.push_back(0);
      ends_indices.push_back(out_dims[axis]);
      strides_indices.push_back(1);
      reverse_axis.push_back(false);
    }
    for (size_t axis = 0; axis < axes_.size(); axis++) {
      int axis_index = axes_[axis];
      starts_indices[axis_index] = starts_[axis];
      ends_indices[axis_index] = ends_[axis];
      strides_indices[axis_index] = strides_[axis];
      reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
    }
    auto out_dims_origin = out_dims;
    if (decrease_axis_.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis_.size(); ++i) {
        out_dims_origin[decrease_axis_[i]] = 0;
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
    for (size_t axis = 0; axis < axes_.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        need_reverse = true;
        break;
      }
    }
    out->Resize(out_dims);
    if (need_reverse) {
      Tensor* tmp = new Tensor();
      tmp->Resize(out_dims);
      auto* tmp_t = tmp->mutable_data<float>();
      stride_slice(input_data,
                   tmp_t,
                   in_dims.data(),
                   out_dims.data(),
                   starts_indices,
                   ends_indices,
                   strides_indices);
      reverse(tmp_t, out_data, out_dims.data(), reverse_axis);
    } else {
      stride_slice(input_data,
                   out_data,
                   in_dims.data(),
                   out_dims.data(),
                   starts_indices,
                   ends_indices,
                   strides_indices);
    }
    if (decrease_axis_.size() > 0) {
      out->Resize(out_dims_origin);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("strided_slice");
    op_desc->SetInput("Input", {input_});

    if (use_tensor_) {
      op_desc->SetInput("StartsTensor", {starts_tensor_});
      op_desc->SetInput("EndsTensor", {ends_tensor_});
    } else if (use_tensor_list_) {
      std::vector<std::string> starts_tensor_list_;
      std::vector<std::string> ends_tensor_list_;
      for (size_t i = 0; i < starts_.size(); ++i) {
        starts_tensor_list_.push_back("starts_tensor_list_" +
                                      paddle::lite::to_string(i));
        ends_tensor_list_.push_back("ends_tensor_list_" +
                                    paddle::lite::to_string(i));
      }
      op_desc->SetInput("StartsTensorList", {starts_tensor_list_});
      op_desc->SetInput("EndsTensorList", {ends_tensor_list_});
    }

    if (infer_flags_.size() > 0) {
      op_desc->SetAttr("infer_flags", infer_flags_);
    }
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axes", axes_);
    op_desc->SetAttr("starts", starts_);
    op_desc->SetAttr("ends", ends_);
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("decrease_axis", decrease_axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i;
    }

    SetCommonTensor(input_, dims_, data.data());
    if (use_tensor_) {
      SetCommonTensor(starts_tensor_,
                      DDim({static_cast<int64_t>(starts_.size())}),
                      starts_i64.data());
      SetCommonTensor(ends_tensor_,
                      DDim({static_cast<int64_t>(ends_.size())}),
                      ends_i64.data());
    } else if (use_tensor_list_) {
      for (size_t i = 0; i < starts_.size(); ++i) {
        SetCommonTensor("starts_tensor_list_" + paddle::lite::to_string(i),
                        DDim({1}),
                        &starts_i64[i]);
      }
      for (size_t i = 0; i < ends_.size(); ++i) {
        SetCommonTensor("ends_tensor_list_" + paddle::lite::to_string(i),
                        DDim({1}),
                        &ends_i64[i]);
      }
    }
  }
};

void test_slice(Place place, float abs_error) {
  std::vector<int> axes({1, 2});
  std::vector<int> starts({2, 2});
  std::vector<int> strides({1, 1});
  std::vector<int> ends({6, 7});
  std::vector<int> infer_flags({1, 1});
  std::vector<int> decrease_axis({});
  DDim dims({10, 10, 10});
  std::unique_ptr<arena::TestCase> tester(
      new StridedSliceComputeTester(place,
                                    "def",
                                    axes,
                                    starts,
                                    ends,
                                    strides,
                                    decrease_axis,
                                    dims,
                                    false,
                                    false,
                                    infer_flags));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_axes(Place place, float abs_error) {
  std::vector<int> axes({1, 2});
  std::vector<int> starts({1, 1});
  std::vector<int> strides({1, 1});
  std::vector<int> ends({2, 3});
  std::vector<int> infer_flags({1, 1});
  std::vector<int> decrease_axis({});
  DDim dims({2, 3, 4, 5});
  std::unique_ptr<arena::TestCase> tester(
      new StridedSliceComputeTester(place,
                                    "def",
                                    axes,
                                    starts,
                                    ends,
                                    strides,
                                    decrease_axis,
                                    dims,
                                    false,
                                    false,
                                    infer_flags));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_decrease_axis(Place place, float abs_error) {
  std::vector<int> axes({1});
  std::vector<int> starts({0});
  std::vector<int> ends({1});
  std::vector<int> strides({1});
  std::vector<int> decrease_axis({1});
  std::vector<int> infer_flags({1});
  DDim dims({2, 3, 4, 5});
  std::unique_ptr<arena::TestCase> tester(
      new StridedSliceComputeTester(place,
                                    "def",
                                    axes,
                                    starts,
                                    ends,
                                    strides,
                                    decrease_axis,
                                    dims,
                                    false,
                                    false,
                                    infer_flags));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_tensor(Place place, float abs_error) {
  std::vector<int> axes({0, 1, 2});
  std::vector<int> starts({2, 2, 2});
  std::vector<int> ends({5, 6, 7});
  std::vector<int> strides({1, 1, 2});
  std::vector<int> infer_flags({1, 1, 1});
  std::vector<int> decrease_axis({});
  DDim dims({10, 10, 10});
  std::unique_ptr<arena::TestCase> tester(
      new StridedSliceComputeTester(place,
                                    "def",
                                    axes,
                                    starts,
                                    ends,
                                    strides,
                                    decrease_axis,
                                    dims,
                                    false,
                                    false,
                                    infer_flags));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_tensor_list(Place place, float abs_error) {
  std::vector<int> axes({0, 1, 2});
  std::vector<int> starts({2, 2, 2});
  std::vector<int> ends({5, 6, 7});
  std::vector<int> strides({1, 1, 2});
  std::vector<int> decrease_axis({});
  std::vector<int> infer_flags({1, 1, 1});
  DDim dims({10, 10, 10});
  std::unique_ptr<arena::TestCase> tester(
      new StridedSliceComputeTester(place,
                                    "def",
                                    axes,
                                    starts,
                                    ends,
                                    strides,
                                    decrease_axis,
                                    dims,
                                    false,
                                    true,
                                    infer_flags));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

TEST(StrideSlice, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  test_slice(place, abs_error);
  test_slice_axes(place, abs_error);
  test_slice_decrease_axis(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  test_slice(place, abs_error);
  test_slice_axes(place, abs_error);
  test_slice_decrease_axis(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 1e-2;
  test_slice(place, abs_error);
  test_slice_axes(place, abs_error);
  test_slice_decrease_axis(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_X86) || defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#endif

  test_slice(place, abs_error);
  test_slice_axes(place, abs_error);
  test_slice_decrease_axis(place, abs_error);
  test_slice_tensor(place, abs_error);
  test_slice_tensor_list(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
