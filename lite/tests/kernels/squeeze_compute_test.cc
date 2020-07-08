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

class SqueezeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::vector<int> axes_;
  DDim dims_;

 public:
  SqueezeComputeTester(const Place& place,
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
    size_t num_squeeze_dims = axes_.size();
    int cnt_squeezed_dims = 0;
    bool should_squeeze[9] = {false};

    if (num_squeeze_dims == 0) {
      for (size_t idx = 0; idx < in_dims.size(); ++idx) {
        if (in_dims[idx] == 1) {
          should_squeeze[idx] = true;
          ++cnt_squeezed_dims;
        }
      }
    } else {
      for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
        int current = axes_[idx] < 0 ? axes_[idx] + in_dims.size() : axes_[idx];
        // Check current index, the upper limit has been checked.
        CHECK_GE(current, 0)
            << "Invalid axis, the negative axis is out of range.";

        CHECK_EQ(in_dims[current], 1) << "Invalid axis index, the axis that "
                                         "will be squeezed should be equal "
                                         "to 1.";
        if (!(should_squeeze[current])) {
          ++cnt_squeezed_dims;
        }
        should_squeeze[current] = true;
      }
    }

    std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
    for (size_t in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
      if (!should_squeeze[in_idx]) {
        output_shape[out_idx++] = in_dims[in_idx];
      }
    }

    out->Resize(DDim(output_shape));
    LOG(INFO) << "baseline out size: " << out->dims();
    auto* input_data = input->data<float>();
    auto* out_data = out->mutable_data<float>();
    memcpy(out_data, input_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("squeeze");
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

class Squeeze2ComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string xshape_ = "XShape";
  std::vector<int> axes_;
  DDim dims_;

 public:
  Squeeze2ComputeTester(const Place& place,
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
    std::vector<int64_t> xshape_sp(dims_.size() + 1, 0);
    for (size_t i = 0; i < dims_.size(); ++i) {
      xshape_sp[i + 1] = dims_[i];
    }
    xshape->Resize(DDim(xshape_sp));

    DDim in_dims(dims_);
    size_t num_squeeze_dims = axes_.size();
    int cnt_squeezed_dims = 0;
    bool should_squeeze[9] = {false};

    if (num_squeeze_dims == 0) {
      for (size_t idx = 0; idx < in_dims.size(); ++idx) {
        if (in_dims[idx] == 1) {
          should_squeeze[idx] = true;
          ++cnt_squeezed_dims;
        }
      }
    } else {
      for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
        int current = axes_[idx] < 0 ? axes_[idx] + in_dims.size() : axes_[idx];
        // Check current index, the upper limit has been checked.
        CHECK_GE(current, 0)
            << "Invalid axis, the negative axis is out of range.";

        CHECK_EQ(in_dims[current], 1) << "Invalid axis index, the axis that "
                                         "will be squeezed should be equal "
                                         "to 1.";
        if (!(should_squeeze[current])) {
          ++cnt_squeezed_dims;
        }
        should_squeeze[current] = true;
      }
    }

    std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
    for (size_t in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
      if (!should_squeeze[in_idx]) {
        output_shape[out_idx++] = in_dims[in_idx];
      }
    }

    out->Resize(DDim(output_shape));

    auto* input_data = input->data<float>();
    auto* out_data = out->mutable_data<float>();
    memcpy(out_data, input_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("squeeze2");
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

void test_squeeze(Place place) {
  for (std::vector<int> axes : {std::vector<int>({}),
                                std::vector<int>({0, 2}),
                                std::vector<int>({0, -2})}) {
    for (int N : {1}) {
      for (int C : {3}) {
        for (int H : {1}) {
          for (int W : {5}) {
            std::unique_ptr<arena::TestCase> tester(new SqueezeComputeTester(
                place, "def", axes, DDim({N, C, H, W})));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

void test_squeeze2(Place place) {
  for (std::vector<int> axes : {std::vector<int>({}),
                                std::vector<int>({0, 2}),
                                std::vector<int>({0, -2})}) {
    for (int N : {1}) {
      for (int C : {3}) {
        for (int H : {1}) {
          for (int W : {5}) {
            std::unique_ptr<arena::TestCase> tester(new Squeeze2ComputeTester(
                place, "def", axes, DDim({N, C, H, W})));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision({"XShape"});
          }
        }
      }
    }
  }
}

TEST(squeeze, precision) {
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  Place place(TARGET(kHost));
#endif
  test_squeeze(place);
}

TEST(squeeze2, precision) {
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  Place place(TARGET(kHost));
#endif
  test_squeeze2(place);
}

}  // namespace lite
}  // namespace paddle
