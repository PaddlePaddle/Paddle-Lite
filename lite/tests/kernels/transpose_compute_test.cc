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

int data_index(std::vector<int> pos, DDimLite dims) {
  int d1 = dims[1];
  int d2 = dims[2];
  int d3 = dims[3];
  return pos[0] * d1 * d2 * d3 + pos[1] * d2 * d3 + pos[2] * d3 + pos[3];
}

std::vector<int> pos_trans(std::vector<int> in_pos, std::vector<int> axis) {
  std::vector<int> out_pos(in_pos.size());
  for (int i = 0; i < axis.size(); i++) {
    out_pos[i] = in_pos[axis[i]];
  }
  return out_pos;
}

class TransposeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "transpose2";
  std::string input_ = "x";
  std::string output_ = "out";
  std::string xshape_ = "xshape";
  DDim dims_;
  std::vector<int> axis_;

 public:
  TransposeComputeTester(const Place& place,
                         const std::string& alias,
                         DDim dims,
                         std::vector<int> axis)
      : TestCase(place, alias), dims_(dims), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    auto* x = scope->FindTensor(input_);

    std::vector<int64_t> out_shape(dims_.size(), 0);
    for (size_t i = 0; i < dims_.size(); i++) {
      out_shape[i] = dims_[axis_[i]];
    }
    out->Resize(out_shape);

    auto y_dims = out->dims();

    int input_n = dims_[0];
    int input_c = dims_[1];
    int input_h = dims_[2];
    int input_w = dims_[3];

    auto input_data = x->data<float>();
    auto output_data = out->mutable_data<float>();

    for (int n = 0; n < input_n; ++n) {
      for (int c = 0; c < input_c; ++c) {
        for (int h = 0; h < input_h; ++h) {
          for (int w = 0; w < input_w; ++w) {
            std::vector<int> in_pos{n, c, h, w};
            std::vector<int> out_pos = pos_trans(in_pos, axis_);
            int in_index = data_index(in_pos, dims_);
            int out_index = data_index(out_pos, y_dims);
            output_data[out_index] = input_data[in_index];
          }
        }
      }
    }

    if (op_type_ == "transpose2") {
      auto* xshape = scope->NewTensor(xshape_);
      auto xshape_dims = dims_.Vectorize();
      xshape_dims.insert(xshape_dims.begin(), 0);
      xshape->Resize(xshape_dims);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    if (op_type_ == "transpose2") {
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

TEST(Transpose, precision) {
  LOG(INFO) << "test Transpose op";
  float abs_error = 2e-5;
  Place place;
#ifdef LITE_WITH_XPU
  place = TARGET(kXPU);
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#else
  return;
#endif

  DDim x_dims{{2, 3, 4, 5}};
  std::vector<std::vector<int>> axes{
      {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {3, 1, 2, 0}, {3, 1, 0, 2}};
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new TransposeComputeTester(place, "def", x_dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

}  // namespace lite
}  // namespace paddle
