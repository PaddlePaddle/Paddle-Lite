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

class WhereComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string y_ = "Y";
  std::string condition_ = "Condition";
  std::string out_ = "out";
  DDim x_dims_{{3, 5, 4, 4}};

 public:
  template <typename T>
  void where_kernel(const operators::WhereParam& param) {
    auto* x = param.x;
    auto* y = param.y;
    auto* condition = param.condition;
    auto* out = param.out;
    auto dims = x->dims();
    auto numel = dims.production();
    const T* x_data = x->template data<T>();
    const T* y_data = y->template data<T>();
    const bool* cond_data = input->data<bool>();
    T* out_data = out->template mutable_data<T>();
    for (int i = 0; i < numel; i++) {
      out_data[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
  }
  WhereComputeTester(const Place& place, const std::string& alias, DDim x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* y = scope->FindTensor(y_);
    auto* condition = scope->FindTensor(condition_);
    auto* out = scope->NewTensor(out_);

    out->Resize(x->dims());
    switch (param.x->precision()) {
      case PRECISION(kFloat):
        where_kernel<float>(param);
        break;
      case PRECISION(kInt32):
        where_kernel<int32_t>(param);
        break;
      case PRECISION(kInt64):
        where_kernel<int64_t>(param);
        break;
      case PRECISION(kInt8):
        where_kernel<int8_t>(param);
        break;
      case PRECISION(kBool):
        where_kernel<bool>(param);
        break;
      default:
        LOG(FATAL) << "Where does not implement for the "
                   << "input type:"
                   << static_cast<int>(param.input->precision());
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("where");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetInput("Condition", {condition_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> dx(x_dims_.production());
    std::vector<float> dy(x_dims_.production());
    fill_data_rand(dx.data(), -1.f, 1.f, x_dims_.production());
    fill_data_rand(dy.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());
    SetCommonTensor(y_, x_dims_, dy.data());
    condition_->Resize(x_dims_);
    condition_->set_precision(PRECISION(kFloat));
    auto data = condition_->mutable_data<bool>();
    for (int i = 0; i < x_dims_.production(); i++) {
      data[i] = (i % 2) ? true : false;
    }
  }
};

void TestWhere(Place place, float abs_error) {
  DDimLite dims{{3, 5, 4, 4}};
  std::unique_ptr<arena::TestCase> tester(
      new WhereComputeTester(place, "def", dims));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(WriteToArray, precision) {
  Place place;
  float abs_error = 1e-5;
#ifdef LITE_WITH_ARM
  place = TARGET(kHost);
#else
  return;
#endif

  TestWhere(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
