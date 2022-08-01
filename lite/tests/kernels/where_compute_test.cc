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
#include "lite/core/test/arena/framework.h"
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
  WhereComputeTester(const Place& place, const std::string& alias, DDim x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* y = scope->FindTensor(y_);
    auto* condition = scope->FindTensor(condition_);
    auto* out = scope->NewTensor(out_);

    out->Resize(x->dims());
    auto numel = x_dims_.production();
    if (x->precision() == PRECISION(kFloat)) {
      const float* x_data = x->template data<float>();
      const float* y_data = y->template data<float>();
      const bool* cond_data = condition->template data<bool>();
      float* out_data = out->template mutable_data<float>();
      for (int i = 0; i < numel; i++) {
        out_data[i] = cond_data[i] ? x_data[i] : y_data[i];
      }
    } else if (x->precision() == PRECISION(kInt32)) {
      const int* x_data1 = x->template data<int>();
      const int* y_data1 = y->template data<int>();
      const bool* cond_data1 = condition->template data<bool>();
      int* out_data1 = out->template mutable_data<int>();
      for (int i = 0; i < numel; i++) {
        out_data1[i] = cond_data1[i] ? x_data1[i] : y_data1[i];
      }
    } else if (x->precision() == PRECISION(kInt64)) {
      const int64_t* x_data2 = x->template data<int64_t>();
      const int64_t* y_data2 = y->template data<int64_t>();
      const bool* cond_data2 = condition->template data<bool>();
      int64_t* out_data2 = out->template mutable_data<int64_t>();
      for (int i = 0; i < numel; i++) {
        out_data2[i] = cond_data2[i] ? x_data2[i] : y_data2[i];
      }
    } else if (x->precision() == PRECISION(kInt8)) {
      const int8_t* x_data3 = x->template data<int8_t>();
      const int8_t* y_data3 = y->template data<int8_t>();
      const bool* cond_data3 = condition->template data<bool>();
      int8_t* out_data3 = out->template mutable_data<int8_t>();
      for (int i = 0; i < numel; i++) {
        out_data3[i] = cond_data3[i] ? x_data3[i] : y_data3[i];
      }
    } else if (x->precision() == PRECISION(kBool)) {
      const bool* x_data4 = x->template data<bool>();
      const bool* y_data4 = y->template data<bool>();
      const bool* cond_data4 = condition->template data<bool>();
      bool* out_data4 = out->template mutable_data<bool>();
      for (int i = 0; i < numel; i++) {
        out_data4[i] = cond_data4[i] ? x_data4[i] : y_data4[i];
      }
    } else {
      LOG(FATAL) << "Where does not implement for the "
                 << "input type:" << static_cast<int>(x->precision());
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
    std::vector<uint8_t> dc(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      dc[i] = (i % 2) ? true : false;
    }
    SetCommonTensor(condition_, x_dims_, reinterpret_cast<bool*>(dc.data()));
  }
};

void TestWhere(Place place, float abs_error) {
  DDimLite x_dims{{3, 5, 4, 4}};
  std::unique_ptr<arena::TestCase> tester(
      new WhereComputeTester(place, "def", x_dims));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(where, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  // TODO(liusiyuan): support later
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestWhere(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
