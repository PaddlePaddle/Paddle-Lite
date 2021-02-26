// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class FlattenComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "flatten";
  std::string input_ = "x";
  std::string output_ = "out";
  std::string xshape_ = "xshape";
  DDim dims_;
  int axis_;

 public:
  FlattenComputeTester(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       int axis)
      : TestCase(place, alias), dims_(dims), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    auto* x = scope->FindTensor(input_);

    int64_t outer = 1, inner = 1;
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (i < axis_) {
        outer *= dims_[i];
      } else {
        inner *= dims_[i];
      }
    }
    std::vector<int64_t> out_shape(2);
    out_shape[0] = outer;
    out_shape[1] = inner;
    out->Resize(out_shape);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();
    memcpy(out_data, x_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    if (op_type_ == "flatten2") {
      op_desc->SetOutput("XShape", {xshape_});
    }
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

void TestFlatten(Place place, float abs_error) {
  DDim dims{{2, 3, 4, 5}};
  std::vector<int> axes{0, 1, 2, 3};
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new FlattenComputeTester(place, "def", dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

TEST(flatten, precision) {
  LOG(INFO) << "test flatten op";
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  TestFlatten(place, abs_error);
}

static std::vector<int64_t> GetOutputShape(const DDim in_dims,
                                           int start_axis,
                                           int stop_axis) {
  int64_t outer = 1;
  std::vector<int64_t> out_shape;
  size_t in_dims_size = in_dims.size();
  out_shape.reserve(in_dims_size - stop_axis + start_axis);

  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(in_dims[i]);
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    outer *= in_dims[i];
  }
  out_shape.push_back(outer);
  for (size_t i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(in_dims[i]);
  }
  return out_shape;
}

class FlattenContiguousRangeTester : public arena::TestCase {
 protected:
  std::string op_type_ = "flatten_contiguous_range";
  std::string input_ = "x";
  std::string output_ = "out";
  std::string xshape_ = "xshape";
  DDim dims_;
  int start_axis_{};
  int stop_axis_{};

 public:
  FlattenContiguousRangeTester(const Place& place,
                               const std::string& alias,
                               DDim dims,
                               int start_axis,
                               int stop_axis)
      : TestCase(place, alias),
        dims_(dims),
        start_axis_(start_axis),
        stop_axis_(stop_axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    auto* x = scope->FindTensor(input_);
    auto out_shape = GetOutputShape(x->dims(), start_axis_, stop_axis_);
    out->Resize(out_shape);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();
    memcpy(out_data, x_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetOutput("XShape", {xshape_});
    op_desc->SetAttr("start_axis", start_axis_);
    op_desc->SetAttr("stop_axis", stop_axis_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

void TestFlattenContiguous(Place place, float abs_error) {
  DDim dims{{2, 3, 4, 5}};
  for (int start_axis : {1, 2}) {
    for (int stop_axis : {2, 3}) {
      std::unique_ptr<arena::TestCase> tester(new FlattenContiguousRangeTester(
          place, "def", dims, start_axis, stop_axis));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision({"xshape"});
    }
  }
}

TEST(flatten_contiguous_range, precision) {
  LOG(INFO) << "test flatten_contiguous_range op";
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestFlattenContiguous(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
