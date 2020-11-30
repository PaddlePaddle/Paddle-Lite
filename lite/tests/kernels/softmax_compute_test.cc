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
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class SoftmaxComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "softmax";
  DDim x_dims_{{1, 2, 3, 4}};
  std::string x_ = "x";
  std::string out_ = "out";
  int axis_ = 1;

 public:
  SoftmaxComputeTest(const Place& place,
                     const std::string& alias,
                     DDim x_dims,
                     int axis)
      : TestCase(place, alias), x_dims_(x_dims), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(x_dims_);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();
    auto x_rank = x_dims_.size();
    if (axis_ < 0) {
      axis_ += x_rank;
    }
    int axis_size = x_dims_[axis_];
    int outer_num = x_dims_.Slice(0, axis_).production();
    int inner_num = x_dims_.Slice(axis_ + 1, x_rank).production();
    int compute_size = outer_num * inner_num;
    for (int i = 0; i < compute_size; i++) {
      int idx_inner = i % inner_num;
      int idx_outer = (i / inner_num) * axis_size;
      int start = idx_outer * inner_num + idx_inner;
      int offset;

      offset = start;
      float max_data = std::numeric_limits<float>::lowest();
      for (int j = 0; j < axis_size; j++) {
        max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
        offset += inner_num;
      }

      offset = start;
      float sum_data = 0.f;
      for (int j = 0; j < axis_size; j++) {
        out_data[offset] = exp(x_data[offset] - max_data);
        sum_data += out_data[offset];
        offset += inner_num;
      }

      offset = start;
      for (int j = 0; j < axis_size; j++) {
        out_data[offset] /= sum_data;
        offset += inner_num;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());
  }
};

TEST(Softmax, precision) {
  LOG(INFO) << "test softmax op";
  float abs_error = 4e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 4e-3;  // Using fp16 in NPU
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 4e-3;  // precision_mode default is force_fp16
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#else
  return;
#endif

  for (auto x_dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {2, 3, 4}, {3, 4}}) {
    int ndims = x_dims.size();
    for (int axis = -1; axis < ndims; axis++) {
#if defined(LITE_WITH_XPU)
      if (axis != -1 && axis != ndims - 1)
        continue;  // -1 and dims.size() - 1 are only supported by XPU
#endif
      std::unique_ptr<arena::TestCase> tester(
          new SoftmaxComputeTest(place, "def", DDim(x_dims), axis));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

}  // namespace lite
}  // namespace paddle
