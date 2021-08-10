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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class SGDComputeTester : public arena::TestCase {
 protected:
  std::string param_ = "param";
  std::string param_out_ = "param_out";
  std::string grad_ = "grad";
  std::string lr_ = "learning_rate";
  float learning_rate_ = 0.01;
  DDim dims_{{2, 5}};

 public:
  SGDComputeTester(const Place& place,
                   const std::string& alias,
                   DDim dims,
                   float learning_rate)
      : TestCase(place, alias), dims_(dims), learning_rate_(learning_rate) {}

  void RunBaseline(Scope* scope) override {
    auto param = scope->FindTensor(param_);
    auto grad = scope->FindTensor(grad_);
    auto lr = scope->FindTensor(lr_);
    auto param_out = scope->NewTensor(param_out_);
    CHECK(param_out);

    auto param_data = param->data<float>();
    auto grad_data = grad->data<float>();
    auto lr_data = *lr->data<float>();

    param_out->Resize(dims_);
    auto param_out_data = param_out->mutable_data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      param_out_data[i] = param_data[i] - lr_data * grad_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sgd");
    op_desc->SetInput("Param", {param_});
    op_desc->SetInput("Grad", {grad_});
    op_desc->SetInput("LearningRate", {lr_});
    op_desc->SetOutput("ParamOut", {param_out_});
  }

  void PrepareData() override {
    std::vector<float> param_data(dims_.production());
    fill_data_rand(param_data.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(param_, dims_, param_data.data());

    std::vector<float> grad_data(dims_.production());
    fill_data_rand(grad_data.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(grad_, dims_, grad_data.data());

    std::vector<float> lr_data(1);
    lr_data[0] = learning_rate_;
    SetCommonTensor(lr_, DDim{{1}}, lr_data.data());
  }
};

TEST(sgd, precision) {
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  std::vector<int64_t> dims{3, 2, 4, 1};
  float lr = 0.01;
  std::unique_ptr<arena::TestCase> tester(
      new SGDComputeTester(place, "def", DDim(dims), lr));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
