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

class ExpandAsComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string target_ = "Target";
  DDim dims_;
  DDim target_dims_;

 public:
  ExpandAsComputeTester(const Place& place,
                        const std::string& alias,
                        DDim dims,
                        DDim target_dims)
      : TestCase(place, alias), dims_(dims), target_dims_(target_dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* input = scope->FindTensor(x_);
    CHECK(input);
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    const auto* target = scope->FindTensor(target_);

    DDim out_shape(input->dims());
    DDim in_shape = input->dims();

    std::vector<int> expand_times_;
    for (size_t i = 0; i < target->dims().size(); ++i) {
      int times = target->dims()[i] / input->dims()[i];
      expand_times_.push_back(times);
    }
    for (size_t i = 0; i < expand_times_.size(); ++i) {
      out_shape[i] *= expand_times_[i];
    }
    out->Resize(out_shape);
    float* out_data = out->mutable_data<float>();
    const float* input_data = input->data<float>();
    std::vector<int> in_stride(in_shape.size(), 1),
        out_stride(out_shape.size(), 1);
    for (int i = in_shape.size() - 2; i >= 0; --i) {
      in_stride[i] = in_shape[i + 1] * in_stride[i + 1];
    }
    for (int i = out_shape.size() - 2; i >= 0; --i) {
      out_stride[i] = out_shape[i + 1] * out_stride[i + 1];
    }
    for (size_t out_id = 0; out_id < out_shape.production(); ++out_id) {
      int in_id = 0;
      for (int i = expand_times_.size() - 1; i >= 0; --i) {
        int in_j = (out_id / out_stride[i]) % in_shape[i];
        in_id += in_j * in_stride[i];
      }
      out_data[out_id] = input_data[in_id];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("expand_as");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Target", {target_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> in_data(dims_.production());
    std::vector<float> target_data(target_dims_.production());
    for (int i = 0; i < dims_.production(); ++i) {
      in_data[i] = i;
    }
    for (int i = 0; i < target_dims_.production(); ++i) {
      target_data[i] = i;
    }
    SetCommonTensor(x_, dims_, in_data.data());
    SetCommonTensor(target_, target_dims_, target_data.data());
  }
};

void test_expand_as_3dim(Place place, float abs_error) {
  for (int C : {3}) {
    for (int H : {2}) {
      for (int W : {4}) {
        std::unique_ptr<arena::TestCase> tester(new ExpandAsComputeTester(
            place, "def", DDim({C, H, W}), DDim({C * 2, H * 3, W * 1})));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void test_expand_as_4dim(Place place, float abs_error) {
  for (int N : {2}) {
    for (int C : {3}) {
      for (int H : {2}) {
        for (int W : {4}) {
          std::unique_ptr<arena::TestCase> tester(
              new ExpandAsComputeTester(place,
                                        "def",
                                        DDim({N, C, H, W}),
                                        DDim({N * 2, C * 3, H * 1, W * 4})));
          arena::Arena arena(std::move(tester), place, abs_error);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(ExpandAs, precision) {
  float abs_error = 1e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  test_expand_as_3dim(place, abs_error);
  test_expand_as_4dim(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
