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

class UnsqueezeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::vector<int> axes_;
  DDim dims_;

 public:
  UnsqueezeComputeTester(const Place& place,
                         const std::string& alias,
                         const std::vector<int>& axes,
                         DDim dims)
      : TestCase(place, alias), axes_(axes), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* input = scope->FindTensor(x_);
    CHECK(input);
    auto* out = scope->NewTensor(out_);
    CHECK(out);

    DDim in_dims(dims_);
    int output_size = in_dims.size() + static_cast<int>(axes_.size());
    int cur_output_size = in_dims.size();
    std::vector<int64_t> output_shape(output_size, 0);

    // Validate Check: rank range.
    CHECK_LE(output_size, 6)
        << "The output tensor's rank should be less than 6.";

    for (int axis : axes_) {
      int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
      // Validate Check: the axis bound
      CHECK((cur >= 0) && (cur <= cur_output_size))
          << "The unsqueeze dims must be within range of current rank.";
      // Move old axis, and insert new axis
      for (int i = cur_output_size; i >= cur; --i) {
        if (output_shape[i] == 1) {
          // Move axis
          output_shape[i + 1] = 1;
          output_shape[i] = 0;
        }
      }

      output_shape[cur] = 1;
      // Add the output size.
      cur_output_size++;
    }

    // Make output shape
    for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
      if (output_shape[out_idx] == 0) {
        output_shape[out_idx] = in_dims[in_idx++];
      }
    }
    for (size_t i = 0; i < output_shape.size(); ++i)
      out->Resize(DDim(output_shape));
    auto* input_data = input->data<float>();
    auto* out_data = out->mutable_data<float>();
    memcpy(out_data, input_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("unsqueeze");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axes", axes_);
  }

  void PrepareData() override {
    std::vector<float> in_data(dims_.production());
    for (int i = 0; i < dims_.production(); ++i) {
      in_data[i] = i;
    }
    SetCommonTensor(x_, dims_, in_data.data());
  }
};

class Unsqueeze2ComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string xshape_ = "XShape";
  std::vector<int> axes_;
  DDim dims_;

 public:
  Unsqueeze2ComputeTester(const Place& place,
                          const std::string& alias,
                          const std::vector<int>& axes,
                          DDim dims)
      : TestCase(place, alias), axes_(axes), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* input = scope->FindTensor(x_);
    CHECK(input);
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    auto* xshape = scope->NewTensor(xshape_);
    CHECK(xshape);
    std::vector<int64_t> xshape_sp(dims_.size() + 1, 1);
    for (size_t i = 0; i < dims_.size(); ++i) {
      xshape_sp[i + 1] = dims_[i];
    }
    xshape->Resize(DDim(xshape_sp));

    DDim in_dims(dims_);
    int output_size = in_dims.size() + static_cast<int>(axes_.size());
    int cur_output_size = in_dims.size();
    std::vector<int64_t> output_shape(output_size, 0);

    // Validate Check: rank range.
    CHECK_LE(output_size, 6)
        << "The output tensor's rank should be less than 6.";

    for (int axis : axes_) {
      int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
      // Validate Check: the axis bound
      CHECK((cur >= 0) && (cur <= cur_output_size))
          << "The unsqueeze dims must be within range of current rank.";
      // Move old axis, and insert new axis
      for (int i = cur_output_size; i >= cur; --i) {
        if (output_shape[i] == 1) {
          // Move axis
          output_shape[i + 1] = 1;
          output_shape[i] = 0;
        }
      }

      output_shape[cur] = 1;
      // Add the output size.
      cur_output_size++;
    }

    // Make output shape
    for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
      if (output_shape[out_idx] == 0) {
        output_shape[out_idx] = in_dims[in_idx++];
      }
    }

    out->Resize(DDim(output_shape));

    auto* input_data = input->data<float>();
    auto* out_data = out->mutable_data<float>();
    auto* xshape_data = xshape->mutable_data<float>();
    memcpy(out_data, input_data, sizeof(float) * dims_.production());
    memcpy(xshape_data, input_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("unsqueeze2");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetOutput("XShape", {xshape_});
    op_desc->SetAttr("axes", axes_);
  }

  void PrepareData() override {
    std::vector<float> in_data(dims_.production());
    for (int i = 0; i < dims_.production(); ++i) {
      in_data[i] = i;
    }
    SetCommonTensor(x_, dims_, in_data.data());
  }
};

void test_unsqueeze(Place place) {
  for (std::vector<int> axes : {std::vector<int>({}),
                                std::vector<int>({0, 2}),
                                std::vector<int>({0, -2})}) {
    for (int N : {1}) {
      for (int C : {3}) {
        for (int H : {1}) {
          for (int W : {5}) {
            std::unique_ptr<arena::TestCase> tester(new UnsqueezeComputeTester(
                place, "def", axes, DDim({N, C, H, W})));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

void test_unsqueeze2(Place place) {
  for (std::vector<int> axes : {std::vector<int>({}),
                                std::vector<int>({0, 2}),
                                std::vector<int>({0, -2})}) {
    for (int N : {1}) {
      for (int C : {3}) {
        for (int H : {1}) {
          for (int W : {5}) {
            std::unique_ptr<arena::TestCase> tester(new Unsqueeze2ComputeTester(
                place, "def", axes, DDim({N, C, H, W})));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(squeeze, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_unsqueeze(place);
#endif
}

TEST(squeeze2, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_unsqueeze2(place);
#endif
}

}  // namespace lite
}  // namespace paddle
