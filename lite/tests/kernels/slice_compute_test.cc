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
static void slice_ref(const float* input,
                      std::vector<int64_t> in_dims,
                      std::vector<int> axes,
                      std::vector<int> starts,
                      std::vector<int> ends,
                      float* out) {
  auto out_dims = in_dims;
  std::vector<int> real_starts(in_dims.size(), 0);
  std::vector<int> real_ends(in_dims.size(), 0);
  std::vector<int> real_step(in_dims.size(), 0);
  for (size_t i = 0; i < in_dims.size(); i++) {
    real_ends[i] = in_dims[i];
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int dim_value = in_dims[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int LEN = in_dims.size();
  std::vector<int> dst_step(LEN);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    dst_step[i] = 1;
  }
  std::vector<int> src_step(LEN);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; i--) {
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
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

class SliceComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "Input";
  std::string output_ = "Out";
  std::vector<int> axes_;
  std::vector<int> starts_;
  std::vector<int64_t> starts_i64;
  std::vector<int64_t> ends_i64;
  std::vector<int> ends_;
  std::vector<int> decrease_axis_;
  DDim dims_;
  std::vector<int> infer_flags_;
  std::string starts_tensor_ = "StartsTensor";
  std::string ends_tensor_ = "EndsTensor";
  // std::string starts_tensor_list_ = "StartsTensorList";
  // std::string ends_tensor_list_ = "EndsTensorList";
  bool use_tensor_;
  bool use_tensor_list_;

 public:
  SliceComputeTester(const Place& place,
                     const std::string& alias,
                     const std::vector<int>& axes,
                     const std::vector<int>& starts,
                     const std::vector<int>& ends,
                     const std::vector<int>& decrease_axis,
                     const DDim& dims,
                     bool use_tensor = false,
                     bool use_tensor_list = false,
                     const std::vector<int>& infer_flags = {})
      : TestCase(place, alias),
        axes_(axes),
        starts_(starts),
        ends_(ends),
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
    auto out_dims = in_dims;
    int dim_value, start, end;

    for (size_t i = 0; i < axes_.size(); ++i) {
      dim_value = out_dims[axes_[i]];
      if (dim_value > 0) {
        start = starts_[i] < 0 ? (starts_[i] + dim_value) : starts_[i];
        end = ends_[i] < 0 ? (ends_[i] + dim_value) : ends_[i];
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        out_dims[axes_[i]] = end - start;
      }
    }

    out->Resize(out_dims);
    auto* out_data = out->mutable_data<float>();
    slice_ref(input_data, in_dims.data(), axes_, starts_, ends_, out_data);

    if (decrease_axis_.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis_.size(); ++i) {
        out_dims[decrease_axis_[i]] = 0;
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
    out->Resize(out_dims);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("slice");
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

void test_slice(Place place) {
  std::vector<int> axes({1, 2});
  std::vector<int> starts({2, 2});
  std::vector<int> ends({6, 7});
  std::vector<int> decrease_axis({});
  DDim dims({10, 10, 10});
  std::unique_ptr<arena::TestCase> tester(new SliceComputeTester(
      place, "def", axes, starts, ends, decrease_axis, dims));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_axes(Place place) {
  std::vector<int> axes({1, 2});
  std::vector<int> starts({1, 1});
  std::vector<int> ends({2, 3});
  std::vector<int> decrease_axis({});
  DDim dims({2, 3, 4, 5});
  std::unique_ptr<arena::TestCase> tester(new SliceComputeTester(
      place, "def", axes, starts, ends, decrease_axis, dims));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_decrease_axis(Place place) {
  std::vector<int> axes({1});
  std::vector<int> starts({0});
  std::vector<int> ends({1});
  std::vector<int> decrease_axis({1});
  DDim dims({2, 3, 4, 5});
  std::unique_ptr<arena::TestCase> tester(new SliceComputeTester(
      place, "def", axes, starts, ends, decrease_axis, dims));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_tensor(Place place) {
  std::vector<int> axes({0, 1, 2});
  std::vector<int> starts({2, 2, 2});
  std::vector<int> ends({5, 6, 7});
  std::vector<int> decrease_axis({});
  DDim dims({10, 10, 10});
  std::unique_ptr<arena::TestCase> tester(new SliceComputeTester(
      place, "def", axes, starts, ends, decrease_axis, dims, true));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

void test_slice_tensor_list(Place place) {
  std::vector<int> axes({0, 1, 2});
  std::vector<int> starts({2, 2, 2});
  std::vector<int> ends({5, 6, 7});
  std::vector<int> decrease_axis({});
  std::vector<int> infer_flags({});
  DDim dims({10, 10, 10});
  std::unique_ptr<arena::TestCase> tester(new SliceComputeTester(place,
                                                                 "def",
                                                                 axes,
                                                                 starts,
                                                                 ends,
                                                                 decrease_axis,
                                                                 dims,
                                                                 false,
                                                                 true,
                                                                 infer_flags));
  arena::Arena arena(std::move(tester), place, 2e-4);
  arena.TestPrecision();
}

TEST(Slice, precision) {
#if defined(LITE_WITH_NNADAPTER)
  Place place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  test_slice(place);
  test_slice_axes(place);
  test_slice_decrease_axis(place);
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  test_slice(place);
  test_slice_axes(place);
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  test_slice(place);
  test_slice_axes(place);
  test_slice_decrease_axis(place);
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  test_slice(place);
  test_slice_axes(place);
  test_slice_decrease_axis(place);
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  test_slice(place);
  test_slice_axes(place);
  test_slice_decrease_axis(place);
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  test_slice(place);
  test_slice_axes(place);
  test_slice_decrease_axis(place);
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  test_slice(place);
  test_slice_axes(place);
  test_slice_decrease_axis(place);
#endif
#elif defined(LITE_WITH_OPENCL)
  Place place = TARGET(kOpenCL);
  test_slice(place);
  test_slice_tensor(place);
  test_slice_tensor_list(place);
#elif defined(LITE_WITH_XPU)
  Place place(TARGET(kXPU));
  test_slice(place);
  test_slice_tensor(place);
  test_slice_tensor_list(place);
#elif defined(LITE_WITH_ARM)
  Place place(TARGET(kARM));
  test_slice(place);
  test_slice_tensor(place);
  test_slice_tensor_list(place);
#elif defined(LITE_WITH_X86)
  Place place(TARGET(kX86));
  test_slice(place);
  test_slice_tensor(place);
  test_slice_tensor_list(place);
#endif
}

}  // namespace lite
}  // namespace paddle
