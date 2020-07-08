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
#include <string>
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
  std::string axes_tensor_ = "AxesTensor";
  std::vector<std::string> axes_tensor_list_;
  std::vector<int> axes_;
  DDim dims_;
  // input_axes_flag_: 1 for axes, 2 for axes_tensor, 3 for axes_tensor_list
  int input_axes_flag_ = 1;

 public:
  UnsqueezeComputeTester(const Place& place,
                         const std::string& alias,
                         const std::vector<int>& axes,
                         DDim dims,
                         int input_axes_flag)
      : TestCase(place, alias), dims_(dims), input_axes_flag_(input_axes_flag) {
    for (int v : axes) {
      axes_.push_back(v);
    }
  }

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
    out->Resize(DDim(output_shape));
    auto* input_data = input->data<float>();
    auto* out_data = out->mutable_data<float>();
    memcpy(out_data, input_data, sizeof(float) * dims_.production());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("unsqueeze");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    if (input_axes_flag_ == 1) {
      op_desc->SetAttr("axes", axes_);
    } else if (input_axes_flag_ == 2) {
      op_desc->SetInput("AxesTensor", {axes_tensor_});
    } else if (input_axes_flag_ == 3) {
      op_desc->SetInput("AxesTensorList", axes_tensor_list_);
    } else {
      LOG(FATAL) << "input input_axes_flag_ error. " << input_axes_flag_;
    }
  }

  void PrepareData() override {
    std::vector<float> in_data(dims_.production());
    for (int i = 0; i < dims_.production(); ++i) {
      in_data[i] = i;
    }
    SetCommonTensor(x_, dims_, in_data.data());

    if (input_axes_flag_ == 2) {
      DDim axes_tensor_dim{{static_cast<int>(axes_.size())}};
      std::vector<int> axes_tensor_data(axes_.size());
      for (int i = 0; i < axes_tensor_dim.production(); i++) {
        axes_tensor_data[i] = axes_[i];
      }
      SetCommonTensor(axes_tensor_, axes_tensor_dim, axes_tensor_data.data());
    } else if (input_axes_flag_ == 3) {
      std::string name = "axes_tensor_";
      for (size_t i = 0; i < axes_.size(); i++) {
        name = name + paddle::lite::to_string(i);
        axes_tensor_list_.push_back(name);
        SetCommonTensor(name, DDim({1}), &axes_[i]);
      }
    }
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
    std::vector<int64_t> xshape_sp(dims_.size() + 1, 0);
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
    memcpy(out_data, input_data, sizeof(float) * dims_.production());
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

void test_unsqueeze(Place place, float abs_error = 2e-5) {
  for (std::vector<int> axes : {std::vector<int>({1}),
                                std::vector<int>({0, 2}),
                                std::vector<int>({0, -2})}) {
    for (auto dims : std::vector<std::vector<int64_t>>{{3}, {3, 5}, {3, 5, 7}})
      for (int input_axes_flag : {1, 2, 3}) {
#ifdef LITE_WITH_NPU
        if (input_axes_flag != 1) continue;
        if (dims.size() + axes.size() > 4) continue;
#endif
        std::unique_ptr<arena::TestCase> tester(new UnsqueezeComputeTester(
            place, "def", axes, DDim(dims), input_axes_flag));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
  }
}

void test_unsqueeze2(Place place, float abs_error = 2e-5) {
  for (std::vector<int> axes : {std::vector<int>({0}),
                                std::vector<int>({0, 2}),
                                std::vector<int>({0, -2})}) {
    for (auto dims :
         std::vector<std::vector<int64_t>>{{3}, {3, 5}, {3, 5, 7}}) {
#ifdef LITE_WITH_NPU
      if (dims.size() + axes.size() > 4) continue;
#endif
      std::unique_ptr<arena::TestCase> tester(
          new Unsqueeze2ComputeTester(place, "def", axes, DDim(dims)));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision({"XShape"});
    }
  }
}

TEST(unsqueeze, precision) {
  Place place;
  float abs_error = 2e-5;
#ifdef LITE_WITH_NPU
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#endif
  test_unsqueeze(place, abs_error);
}

TEST(unsqueeze2, precision) {
  Place place;
  float abs_error = 2e-5;
#ifdef LITE_WITH_NPU
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#endif

  test_unsqueeze2(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
