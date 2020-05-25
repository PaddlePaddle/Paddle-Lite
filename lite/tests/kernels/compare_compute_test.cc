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

#define COMPARE_FUNCTOR(name, op)                                           \
  template <typename T>                                                     \
  struct name##Functor {                                                    \
    inline bool operator()(const T& a, const T& b) const { return a op b; } \
  };

COMPARE_FUNCTOR(Equal, ==);
COMPARE_FUNCTOR(NotEqual, !=);
COMPARE_FUNCTOR(LessThan, <);
COMPARE_FUNCTOR(LessEqual, <=);
COMPARE_FUNCTOR(GreaterThan, >);
COMPARE_FUNCTOR(GreaterEqual, >=);

template <>
struct EqualFunctor<float> {
  inline bool operator()(const float& a, const float& b) const {
    // It is safe to cast a and b to double.
    return fabs(static_cast<double>(a - b)) < 1e-8;
  }
};

template <>
struct NotEqualFunctor<float> {
  inline bool operator()(const float& a, const float& b) const {
    return !EqualFunctor<float>()(a, b);
  }
};

template <typename T, template <typename U> class Functor>
class CompareComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string y_ = "y";
  std::string out_ = "out";
  std::string op_ = "less_than";
  DDim x_dims_{{3, 5, 4, 4}};
  DDim y_dims_{{4}};
  int axis_ = -1;
  bool force_cpu_ = false;

 public:
  CompareComputeTester(const Place& place,
                       const std::string& alias,
                       const std::string op,
                       DDim x_dims,
                       DDim y_dims,
                       int axis = -1)
      : TestCase(place, alias),
        op_(op),
        x_dims_(x_dims),
        y_dims_(y_dims),
        axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(x_dims_);
    auto* out_data = out->template mutable_data<bool>();
    auto axis = axis_;
    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->template data<T>();
    auto* y = scope->FindTensor(y_);
    auto* y_data_in = y->template data<T>();

    using CompareFunc = Functor<T>;
    if (x_dims_.size() == y_dims_.size()) {
      for (int i = 0; i < x_dims_.production(); i++) {
        out_data[i] = CompareFunc()(x_data[i], y_data_in[i]);
      }
    } else {
      auto* y_data =
          reinterpret_cast<T*>(malloc(x_dims_.production() * sizeof(T)));

      if (axis < 0) {
        axis = x_dims_.size() - y_dims_.size();
      }
      int batch = 1;
      int channels = 1;
      int num = 1;
      for (int i = 0; i < axis; ++i) {
        batch *= x_dims_[i];
      }
      for (int i = 0; i < y_dims_.size(); ++i) {
        channels *= y_dims_[i];
      }
      for (int i = y_dims_.size() + axis; i < x_dims_.size(); ++i) {
        num *= x_dims_[i];
      }
      int ysize = channels * num;
      T* y_data_t = reinterpret_cast<T*>(y_data);
      if (num == 1) {
        for (int i = 0; i < batch; ++i) {
          memcpy(reinterpret_cast<void*>(y_data_t),
                 reinterpret_cast<const void*>(&y_data_in[0]),
                 ysize * sizeof(T));
          y_data_t += ysize;
        }

      } else {
        for (int i = 0; i < channels; i++) {
          for (int j = 0; j < num; j++) {
            y_data_t[i * num + j] = y_data_in[i];
          }
        }
        T* tempptr = y_data_t;
        for (int i = 0; i < batch; ++i) {
          memcpy(y_data_t, tempptr, ysize * sizeof(T));
          y_data_t += ysize;
        }
      }
      for (int i = 0; i < x_dims_.production(); i++) {
        out_data[i] = CompareFunc()(x_data[i], y_data[i]);
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("force_cpu", force_cpu_);
  }

  void PrepareData() override {
    std::vector<T> dx(x_dims_.production());
    std::vector<T> dy(y_dims_.production());
    fill_data_rand<T>(dx.data(), -5, 5, x_dims_.production());
    fill_data_rand<T>(dy.data(), -5, 5, y_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());
    SetCommonTensor(y_, y_dims_, dy.data());
  }
};

template <typename T>
void TestCompare(Place place,
                 float abs_error,
                 std::string op,
                 std::vector<int64_t> x_dims,
                 std::vector<int64_t> y_dims,
                 int axis) {
  if (typeid(T) == typeid(float)) {
    place.precision = PRECISION(kFloat);
  } else if (typeid(T) == typeid(int32_t)) {
    place.precision = PRECISION(kInt32);
  } else if (typeid(T) == typeid(int64_t)) {
    place.precision = PRECISION(kInt64);
  } else {
    LOG(FATAL) << "unsupported dtype";
  }

  std::unique_ptr<arena::TestCase> tester = nullptr;
  if (op == "equal") {
    tester = static_cast<std::unique_ptr<arena::TestCase>>(
        new CompareComputeTester<T, EqualFunctor>(
            place, "def", op, DDim(x_dims), DDim(y_dims), axis));
  } else if (op == "not_equal") {
    tester = static_cast<std::unique_ptr<arena::TestCase>>(
        new CompareComputeTester<T, NotEqualFunctor>(
            place, "def", op, DDim(x_dims), DDim(y_dims), axis));
  } else if (op == "less_than") {
    tester = static_cast<std::unique_ptr<arena::TestCase>>(
        new CompareComputeTester<T, LessThanFunctor>(
            place, "def", op, DDim(x_dims), DDim(y_dims), axis));
  } else if (op == "less_equal") {
    tester = static_cast<std::unique_ptr<arena::TestCase>>(
        new CompareComputeTester<T, LessEqualFunctor>(
            place, "def", op, DDim(x_dims), DDim(y_dims), axis));
  } else if (op == "greater_than") {
    tester = static_cast<std::unique_ptr<arena::TestCase>>(
        new CompareComputeTester<T, GreaterThanFunctor>(
            place, "def", op, DDim(x_dims), DDim(y_dims), axis));
  } else if (op == "greater_equal") {
    tester = static_cast<std::unique_ptr<arena::TestCase>>(
        new CompareComputeTester<T, GreaterEqualFunctor>(
            place, "def", op, DDim(x_dims), DDim(y_dims), axis));
  } else {
    LOG(FATAL) << "unsupported type";
  }
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

#if defined(LITE_WITH_NPU)
TEST(Compare_OP_NPU, precision) {
  Place place{TARGET(kNPU)};
  float abs_error = 1e-2;

  TestCompare<float>(
      place, abs_error, "less_than", {2, 3, 4, 5}, {2, 3, 4, 5}, -1);
  TestCompare<float>(place, abs_error, "less_than", {2, 3, 4}, {2, 3, 4}, 0);
}
#elif defined(LITE_WITH_ARM)
TEST(Compare_OP_ARM, precision) {
  Place place{TARGET(kHost)};
  float abs_error = 1e-5;
  for (auto op : std::vector<std::string>{"equal",
                                          "not_equal",
                                          "less_than",
                                          "less_equal",
                                          "greater_than",
                                          "greater_equal"}) {
    TestCompare<float>(place, abs_error, op, {2, 3, 4, 5}, {2, 3, 4, 5}, -1);
    TestCompare<float>(place, abs_error, op, {2, 3, 4}, {2, 3, 4}, 0);
  }

  TestCompare<float>(place, abs_error, "equal", {2, 3, 4}, {3, 4}, 1);
  TestCompare<float>(place, abs_error, "equal", {2, 3, 4, 5}, {3, 4}, 1);
  TestCompare<float>(place, abs_error, "equal", {2, 3, 4}, {4}, 2);
  TestCompare<float>(place, abs_error, "equal", {2, 3, 4, 5}, {5}, 3);

  TestCompare<int32_t>(place, abs_error, "less_than", {3, 4}, {3, 4}, -1);
  TestCompare<int64_t>(place, abs_error, "less_than", {3, 4}, {3, 4}, -1);
}
#endif

}  // namespace lite
}  // namespace paddle
