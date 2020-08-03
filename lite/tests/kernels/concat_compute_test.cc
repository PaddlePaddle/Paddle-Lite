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

namespace paddle {
namespace lite {

DDim infer_shape(const std::vector<const Tensor*>& inputs, int in_axis) {
  std::vector<DDim> input_dims;
  for (auto* tensor : inputs) {
    input_dims.push_back(tensor->dims());
  }
  size_t axis = static_cast<size_t>(in_axis);

  DDim out_dims = input_dims[0];
  for (size_t i = 1; i < input_dims.size(); i++) {
    for (size_t j = 0; j < input_dims[0].size(); j++) {
      if (j == axis) {
        out_dims[axis] += input_dims[i][j];
      } else {
        if (out_dims[j] != input_dims[i][j]) {
          LOG(FATAL) << "infer shape error.";
        }
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }

  return out_dims;
}

class ConcateComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::vector<std::string> x_vct_{};
  std::string out_ = "out";
  std::string axis_tensor_ = "axis_tensor";
  int axis_ = 0;
  bool is_use_axis_tensor_ = false;

  int x_num_ = 3;
  DDim x_dims_{{2, 3, 4, 5}};

 public:
  ConcateComputeTester(const Place& place,
                       const std::string& alias,
                       int axis,
                       bool is_use_axis_tensor)
      : TestCase(place, alias) {
    axis_ = axis;
    is_use_axis_tensor_ = is_use_axis_tensor;
  }

  void RunBaseline(Scope* scope) override {
    std::vector<const Tensor*> x_vct;
    for (std::string& name : x_vct_) {
      x_vct.push_back(scope->FindTensor(name));
    }

    auto* out = scope->NewTensor(out_);
    DDim output_dims = infer_shape(x_vct, axis_);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();

    int num = x_vct.size();
    int rows = 1;
    auto dim_0 = x_vct[0]->dims();
    for (int i = 0; i < axis_; ++i) {
      rows *= dim_0[i];
    }
    int out_rows = rows, out_cols = 0;

    std::vector<int> input_cols(x_vct.size());
    for (int i = 0; i < num; ++i) {
      int input_i_numel = x_vct[i]->dims().size() == 0 ? 0 : 1;
      for (int didx = 0; didx < x_vct[i]->dims().size(); ++didx) {
        input_i_numel *= x_vct[i]->dims()[didx];
      }
      int t_cols = input_i_numel / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }

    // computation
    int col_idx = 0;
    for (int j = 0; j < num; ++j) {
      int col_len = input_cols[j];
      auto input_data = x_vct[j]->data<float>();
      for (int k = 0; k < out_rows; ++k) {
        memcpy(output_data + k * out_cols + col_idx,
               input_data + k * col_len,
               sizeof(float) * col_len);
      }
      col_idx += col_len;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("concat");
    op_desc->SetInput("X", x_vct_);
    op_desc->SetAttr("axis", axis_);
    if (is_use_axis_tensor_) {
      op_desc->SetInput("AxisTensor", {axis_tensor_});
    }
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    for (int n = 0; n < x_num_; n++) {
      std::vector<float> x_data(x_dims_.production());
      for (int i = 0; i < x_dims_.production(); i++) {
        x_data[i] = static_cast<float>(i + n);
      }
      const std::string x_name = "x_tensor_" + paddle::lite::to_string(n);
      x_vct_.push_back(x_name);
      SetCommonTensor(x_name, x_dims_, x_data.data());
    }

    if (is_use_axis_tensor_) {
      SetCommonTensor(axis_tensor_, DDim({1}), &axis_);
      LOG(INFO) << "set axis tensor";
    }
  }
};

TEST(Concat, precision) {
  LOG(INFO) << "test concat op, kARM";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (int axis : {1, 2}) {
    for (bool is_use_axis_tensor : {false, true}) {
      // is_use_axis_tensor = true has bugs in Huawei Ascend NPU DDK
      if (place == TARGET(kHuaweiAscendNPU) && is_use_axis_tensor) {
        continue;
      }
      LOG(INFO) << "axis:" << axis
                << ", is_use_axis_tensor:" << is_use_axis_tensor;
      std::unique_ptr<arena::TestCase> tester(
          new ConcateComputeTester(place, "def", axis, is_use_axis_tensor));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

}  // namespace lite
}  // namespace paddle
