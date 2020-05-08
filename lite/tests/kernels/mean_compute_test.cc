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

class MeanComputeTester : public arena::TestCase {
 protected:
  DDim input_dims_{{2, 5}};
  std::string input_ = "x";
  std::string output_ = "out";

 public:
  MeanComputeTester(const Place& place,
                    const std::string& alias,
                    const DDim& input_dims)
      : TestCase(place, alias), input_dims_(input_dims) {}

  void RunBaseline(Scope* scope) override {
    auto input = scope->FindTensor(input_);
    auto output = scope->NewTensor(output_);

    std::vector<int64_t> out_dims{1};
    output->Resize(out_dims);

    auto input_data = input->data<float>();
    auto output_data = output->mutable_data<float>();

    int x_size = input_dims_.production();
    float sum = 0;
    for (int i = 0; i < x_size; i++) {
      sum += input_data[i];
    }
    output_data[0] = sum / x_size;
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("mean");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<float> input(input_dims_.production());
    fill_data_rand(input.data(), -1.f, 1.f, input_dims_.production());
    SetCommonTensor(input_, input_dims_, input.data());
  }
};

void TestNormalCase(Place place, float abs_error = 2e-5) {
  LOG(INFO) << "Test Mean";
  for (std::vector<int64_t> dims : std::vector<std::vector<int64_t>>{
           {5}, {4, 5}, {3, 4, 5}, {2, 3, 4, 5}}) {
    std::unique_ptr<arena::TestCase> tester(
        new MeanComputeTester(place, "def", DDim(dims)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

#ifdef LITE_WITH_TRAIN
class MeanGradComputeTester : public arena::TestCase {
 protected:
  DDim input_dims_{{2, 5}};
  DDim output_grad_dims_{{1}};
  std::string input_ = "x";
  std::string input_grad_ = "x_grad";
  std::string output_grad_ = "out_grad";

 public:
  MeanGradComputeTester(const Place& place,
                        const std::string& alias,
                        const DDim& input_dims)
      : TestCase(place, alias), input_dims_(input_dims) {}

  void RunBaseline(Scope* scope) override {
    auto input = scope->FindTensor(input_);
    auto output_grad = scope->FindTensor(output_grad_);
    auto input_grad = scope->NewTensor(input_grad_);

    input_grad->Resize(input_dims_);

    auto input_data = input->data<float>();
    auto output_grad_data = output_grad->data<float>();
    auto input_grad_data = input_grad->mutable_data<float>();

    int x_size = input_dims_.production();
    float d_x = output_grad_data[0] / x_size;

    for (int i = 0; i < x_size; i++) {
      input_grad_data[i] = d_x;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("mean_grad");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Out@GRAD", {output_grad_});
    op_desc->SetOutput("X@GRAD", {input_grad_});
  }

  void PrepareData() override {
    std::vector<float> input(input_dims_.production());
    fill_data_rand(input.data(), -1.f, 1.f, input_dims_.production());
    SetCommonTensor(input_, input_dims_, input.data());

    std::vector<float> output_grad(1);
    fill_data_rand(output_grad.data(), -1.f, 1.f, 1);
    SetCommonTensor(output_grad_, output_grad_dims_, output_grad.data());
  }
};

void TestGradNormalCase(Place place, float abs_error = 2e-5) {
  LOG(INFO) << "Test Mean Grad";
  for (std::vector<int64_t> dims : std::vector<std::vector<int64_t>>{
           {5}, {4, 5}, {3, 4, 5}, {2, 3, 4, 5}}) {
    std::unique_ptr<arena::TestCase> tester(
        new MeanGradComputeTester(place, "def", DDim(dims)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}
#endif

TEST(Mean, precision) {
#ifdef LITE_WITH_ARM
  float abs_error = 2e-5;
  Place place(TARGET(kARM));
  TestNormalCase(place, abs_error);
#ifdef LITE_WITH_TRAIN
  TestGradNormalCase(place, abs_error);
#endif
#endif
}

}  // namespace lite
}  // namespace paddle
