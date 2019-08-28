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

class ElementwiseComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  ElementwiseComputeTester(const Place& place,
                           const std::string& alias,
                           int axis)
      : TestCase(place, alias), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] + y_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("elementwise_add");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data.data());
  }
};

class ElementwiseMulComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  ElementwiseMulComputeTester(const Place& place,
                              const std::string& alias,
                              int axis)
      : TestCase(place, alias), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] * y_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("elementwise_mul");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data.data());
  }
};

class ElementwiseMaxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  ElementwiseMaxComputeTester(const Place& place,
                              const std::string& alias,
                              int axis)
      : TestCase(place, alias), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = std::max(x_data[i], y_data[i]);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("elementwise_max");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data.data());
  }
};

class FusionElementwiseAddActivationComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  std::string act_type_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  FusionElementwiseAddActivationComputeTester(const Place& place,
                                              const std::string& alias,
                                              int axis,
                                              std::string act_type)
      : TestCase(place, alias), axis_(axis), act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] + y_data[i];
      if (act_type_ == "relu") {
        out_data[i] = out_data[i] > 0 ? out_data[i] : 0;
      } else {
        LOG(FATAL) << "unsupported Activation type: " << act_type_;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fusion_elementwise_add_activation");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("act_type", act_type_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data.data());
  }
};

class FusionElementwiseMulActivationComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  std::string act_type_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  FusionElementwiseMulActivationComputeTester(const Place& place,
                                              const std::string& alias,
                                              int axis,
                                              std::string act_type)
      : TestCase(place, alias), axis_(axis), act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] * y_data[i];
      if (act_type_ == "relu") {
        out_data[i] = out_data[i] > 0 ? out_data[i] : 0;
      } else {
        LOG(FATAL) << "unsupported Activation type: " << act_type_;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fusion_elementwise_mul_activation");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("act_type", act_type_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data.data());
  }
};

class FusionElementwiseMaxActivationComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  std::string act_type_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  FusionElementwiseMaxActivationComputeTester(const Place& place,
                                              const std::string& alias,
                                              int axis,
                                              std::string act_type)
      : TestCase(place, alias), axis_(axis), act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = x->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = std::max(x_data[i], y_data[i]);
      if (act_type_ == "relu") {
        out_data[i] = out_data[i] > 0 ? out_data[i] : 0;
      } else {
        LOG(FATAL) << "unsupported Activation type: " << act_type_;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fusion_elementwise_max_activation");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("act_type", act_type_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data.data());
  }
};

class ElementwiseDivComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  ElementwiseDivComputeTester(const Place& place,
                              const std::string& alias,
                              int axis)
      : TestCase(place, alias), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = y->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] / y_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("elementwise_div");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    std::vector<float> data2(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      data2[i] = (i + 1) * 1.1;
    }

    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data2.data());
  }
};

class FusionElementwiseDivActivationComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string inputx_ = "x";
  std::string inputy_ = "y";
  std::string output_ = "out";
  int axis_;
  std::string act_type_;
  DDim dims_{{1, 2, 3, 4}};

 public:
  FusionElementwiseDivActivationComputeTester(const Place& place,
                                              const std::string& alias,
                                              int axis,
                                              std::string act_type)
      : TestCase(place, alias), axis_(axis), act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(inputx_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(inputy_);
    const auto* y_data = y->data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = x_data[i] / y_data[i];
      if (act_type_ == "relu") {
        out_data[i] = out_data[i] > 0 ? out_data[i] : 0;
      } else {
        LOG(FATAL) << "unsupported Activation type: " << act_type_;
      }
      LOG(INFO) << "fusion div resul:" << out_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fusion_elementwise_div_activation");
    op_desc->SetInput("X", {inputx_});
    op_desc->SetInput("Y", {inputy_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("act_type", act_type_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }
    std::vector<float> data2(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      data2[i] = (i + 1) * 1.1;
    }
    SetCommonTensor(inputx_, dims_, data.data());
    SetCommonTensor(inputy_, dims_, data2.data());
  }
};

void test_elementwise(Place place) {
  for (int axis : {-1, 0, 1, 3}) {
    std::unique_ptr<arena::TestCase> tester(
        new ElementwiseComputeTester(place, "def", axis));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();

    std::unique_ptr<arena::TestCase> tester_mul(
        new ElementwiseMulComputeTester(place, "def", axis));
    arena::Arena arena_mul(std::move(tester_mul), place, 2e-5);
    arena_mul.TestPrecision();

    std::unique_ptr<arena::TestCase> tester_max(
        new ElementwiseMaxComputeTester(place, "def", axis));
    arena::Arena arena_max(std::move(tester_max), place, 2e-5);
    arena_max.TestPrecision();

    std::unique_ptr<arena::TestCase> tester_div(
        new ElementwiseDivComputeTester(place, "def", axis));
    arena::Arena arena_div(std::move(tester_div), place, 2e-5);
    arena_div.TestPrecision();
  }
}

TEST(Elementwise, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_elementwise(place);
#endif
}

void test_fusion_elementwise(Place place) {
  for (int axis : {-1, 0, 1, 3}) {
    std::unique_ptr<arena::TestCase> tester_add_act(
        new FusionElementwiseAddActivationComputeTester(
            place, "def", axis, "relu"));
    arena::Arena arena_add_act(std::move(tester_add_act), place, 2e-5);
    arena_add_act.TestPrecision();

    std::unique_ptr<arena::TestCase> tester_mul_act(
        new FusionElementwiseMulActivationComputeTester(
            place, "def", axis, "relu"));
    arena::Arena arena_mul_act(std::move(tester_mul_act), place, 2e-5);
    arena_mul_act.TestPrecision();

    std::unique_ptr<arena::TestCase> tester_max_act(
        new FusionElementwiseMaxActivationComputeTester(
            place, "def", axis, "relu"));
    arena::Arena arena_max_act(std::move(tester_max_act), place, 2e-5);
    arena_max_act.TestPrecision();

    std::unique_ptr<arena::TestCase> tester_div_act(
        new FusionElementwiseDivActivationComputeTester(
            place, "def", axis, "relu"));
    arena::Arena arena_div_act(std::move(tester_div_act), place, 2e-5);
    arena_div_act.TestPrecision();
  }
}

TEST(FusionElementwise, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_fusion_elementwise(place);
#endif
}

}  // namespace lite
}  // namespace paddle
