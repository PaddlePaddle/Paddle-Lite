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

struct _logical_and_func {
  inline bool operator()(const bool& a, const bool& b) const { return a && b; }
};

struct _logical_or_func {
  inline bool operator()(const bool& a, const bool& b) const { return a || b; }
};

struct _logical_xor_func {
  inline bool operator()(const bool& a, const bool& b) const {
    return (a || b) && !(a && b);
  }
};

struct _logical_not_func {
  inline bool operator()(const bool& a, const bool& b) const { return !a; }
};

template <class Functor>
class LogicalTester : public arena::TestCase {
 protected:
  std::string op_type_ = "logical_xor";
  std::string x_ = "x";
  std::string y_ = "y";
  std::string out_ = "out";
  DDim dims_{{2, 3, 4, 5}};

 public:
  LogicalTester(const Place& place,
                const std::string& alias,
                const std::string& op_type)
      : TestCase(place, alias), op_type_(op_type) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    const bool* x_data = x->template data<bool>();
    const Tensor* y = nullptr;
    const bool* y_data = nullptr;
    if (op_type_ != "logical_not") {
      y = scope->FindTensor(y_);
      y_data = y->template data<bool>();
    }

    auto* out = scope->NewTensor(out_);
    out->Resize(dims_);
    bool* out_data = out->template mutable_data<bool>();
    for (int i = 0; i < dims_.production(); i++) {
      bool y_tmp = (y_data == nullptr) ? true : y_data[i];
      out_data[i] = Functor()(x_data[i], y_tmp);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    if (op_type_ != "logical_not") {
      op_desc->SetInput("Y", {y_});
    }
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    bool* dx = new bool[dims_.production()];
    for (int64_t i = 0; i < dims_.production(); i++) {
      dx[i] = (i % 3 == 0);
    }
    SetCommonTensor(x_, dims_, dx);
    delete dx;

    if (op_type_ != "logical_not") {
      bool* dy = new bool[dims_.production()];
      for (int64_t i = 0; i < dims_.production(); i++) {
        dy[i] = (i % 2 == 0);
      }
      SetCommonTensor(y_, dims_, dy);
      delete dy;
    }
  }
};

void TestLogical(Place place, float abs_error) {
  std::unique_ptr<arena::TestCase> logical_and_tester(
      new LogicalTester<_logical_and_func>(place, "def", "logical_and"));
  arena::Arena arena_and(std::move(logical_and_tester), place, abs_error);
  arena_and.TestPrecision();

  std::unique_ptr<arena::TestCase> logical_or_tester(
      new LogicalTester<_logical_or_func>(place, "def", "logical_or"));
  arena::Arena arena_or(std::move(logical_or_tester), place, abs_error);
  arena_or.TestPrecision();

  std::unique_ptr<arena::TestCase> logical_xor_tester(
      new LogicalTester<_logical_xor_func>(place, "def", "logical_xor"));
  arena::Arena arena_xor(std::move(logical_xor_tester), place, abs_error);
  arena_xor.TestPrecision();

  std::unique_ptr<arena::TestCase> logical_not_tester(
      new LogicalTester<_logical_not_func>(place, "def", "logical_not"));
  arena::Arena arena_not(std::move(logical_not_tester), place, abs_error);
  arena_not.TestPrecision();
}

TEST(Logical, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestLogical(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
