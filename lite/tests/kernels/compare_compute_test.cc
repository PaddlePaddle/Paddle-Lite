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
#define COMPARE_FUNCTOR(name, op)                                           \
  template <typename T>                                                     \
  struct _##name##Functor {                                                 \
    inline bool operator()(const T& a, const T& b) const { return a op b; } \
  };

COMPARE_FUNCTOR(Equal, ==);
COMPARE_FUNCTOR(NotEqual, !=);
COMPARE_FUNCTOR(LessThan, <);
COMPARE_FUNCTOR(LessEqual, <=);
COMPARE_FUNCTOR(GreaterThan, >);
COMPARE_FUNCTOR(GreaterEqual, >=);

template <>
struct _EqualFunctor<float> {
  inline bool operator()(const float& a, const float& b) const {
    // It is safe to cast a and b to double.
    return fabs(static_cast<double>(a - b)) < 1e-8;
  }
};

template <>
struct _NotEqualFunctor<float> {
  inline bool operator()(const float& a, const float& b) const {
    return !_EqualFunctor<float>()(a, b);
  }
};

template <template <typename T> class Functor>
class LessThanTester : public arena::TestCase {
 protected:
  std::string input_x_ = "x";
  std::string input_y_ = "y";
  std::string output_ = "out";
  int axis_ = 1;
  bool force_cpu_ = 0;
  DDim x_dims_{{3, 5, 4, 4}};
  DDim y_dims_{{4}};
  std::string opname_ = "less_than";

 public:
  LessThanTester(const Place& place,
                 const std::string& alias,
                 bool force_cpu,
                 int axis,
                 DDim x_dims,
                 DDim y_dims,
                 const std::string& opname)
      : TestCase(place, alias),
        axis_(axis),
        force_cpu_(force_cpu),
        x_dims_(x_dims),
        y_dims_(y_dims),
        opname_(opname) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(x_dims_);
    auto* out_data = out->mutable_data<bool>();
    auto axis = axis_;
    auto* x = scope->FindTensor(input_x_);
    const auto* x_data = x->data<float>();
    auto* y = scope->FindTensor(input_y_);
    const auto* y_data_in = y->data<float>();

    auto* y_data = y_data_in;
    if (x_dims_.size() != y_dims_.size()) {
      y_data =
          reinterpret_cast<float*> malloc(x_dims_.production() * sizeof(float));

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
      float* y_data_t = reinterpret_cast<float*> y_data;
      if (num == 1) {
        for (int i = 0; i < batch; ++i) {
          memcpy(y_data_t,
                 reinterpret_cast<void*>(&y_data_in[0]),
                 ysize * sizeof(float));
          y_data_t += ysize;
        }

      } else {
        for (int i = 0; i < channels; i++) {
          for (int j = 0; j < num; j++) {
            y_data_t[i * num + j] = y_data_in[i];
          }
        }
        float* tempptr = y_data_t;
        for (int i = 0; i < batch; ++i) {
          memcpy(y_data_t, tempptr, ysize * sizeof(float));
          y_data_t += ysize;
        }
      }
    }
    using CompareFunc = Functor<float>;
    for (int i = 0; i < x_dims_.production(); i++) {
      // out_data[i] = x_data[i] < y_data[i];
      out_data[i] = CompareFunc()(x_data[i], y_data[i]);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(opname_);
    op_desc->SetInput("X", {input_x_});
    op_desc->SetInput("Y", {input_y_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("force_cpu", force_cpu_);
  }

  void PrepareData() override {
    std::vector<float> data(x_dims_.production());
    std::vector<float> datay(
        y_dims_.production());  // datay(dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      data[i] = 1.1;
    }
    for (int i = 0; i < y_dims_.production(); i++) {
      datay[i] = i;
    }
    SetCommonTensor(input_x_, x_dims_, data.data());
    SetCommonTensor(input_y_, y_dims_, datay.data());
  }
};
void test_compare(Place place) {
  for (bool force_cpu : {0}) {
    for (auto n : {1, 3, 4}) {
      for (auto c : {1, 3, 4}) {
        for (auto h : {1, 3, 4}) {
          for (auto w : {1, 3, 4}) {
            for (auto axis : {-1, 0, 1, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({n, c, h, w})}) {
                DDimLite x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
                DDimLite y_dims = DDim(yd);
                int axis_t = axis < 0 ? x_dims.size() - y_dims.size() : axis;

                if (axis_t + y_dims.size() > 4) continue;
                bool flag = false;
                for (int i = 0; i < y_dims.size(); i++) {
                  if (x_dims[i + axis_t] != y_dims[i]) flag = true;
                }
                if (flag) continue;
                std::unique_ptr<arena::TestCase> less_than_tester(
                    new LessThanTester<paddle::lite::_LessThanFunctor>(
                        place,
                        "def",
                        force_cpu,
                        axis,
                        x_dims,
                        y_dims,
                        "less_than"));
                arena::Arena less_than_arena(
                    std::move(less_than_tester), place, 0.001);
                less_than_arena.TestPrecision();
                std::unique_ptr<arena::TestCase> equal_tester(
                    new LessThanTester<paddle::lite::_EqualFunctor>(place,
                                                                    "def",
                                                                    force_cpu,
                                                                    axis,
                                                                    x_dims,
                                                                    y_dims,
                                                                    "equal"));
                arena::Arena equal_arena(std::move(equal_tester), place, 0.001);
                equal_arena.TestPrecision();
                std::unique_ptr<arena::TestCase> greater_than_tester(
                    new LessThanTester<paddle::lite::_GreaterThanFunctor>(
                        place,
                        "def",
                        force_cpu,
                        axis,
                        x_dims,
                        y_dims,
                        "greater_than"));
                arena::Arena greater_than_arena(
                    std::move(greater_than_tester), place, 0.001);
                greater_than_arena.TestPrecision();
              }
            }
          }
        }
      }
    }
  }
}
TEST(Compare_OP, precision) {
// #ifdef LITE_WITH_X86
// //   Place place(TARGET(kX86));
// // #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_compare(place);
#endif
}

}  // namespace lite
}  // namespace paddle
