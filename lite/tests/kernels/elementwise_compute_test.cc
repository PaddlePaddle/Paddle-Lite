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
#include <cmath>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
// #include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

// int randint(int beg, int end) {
//  int res = 0;
//  fill_data_rand<int>(&res, beg, end, 1);
//  return res;
//}

#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
#endif

#define ELT(MATHOP)                                                          \
  for (int n = 0; n < xn; n++) {                                             \
    for (int c = 0; c < xc; c++) {                                           \
      for (int h = 0; h < xh; h++) {                                         \
        for (int w = 0; w < xw; w++) {                                       \
          int x_offset = n * xc * xh * xw + c * xh * xw + h * xw + w;        \
          int y_offset = 0;                                                  \
          if (yn != 1) y_offset += n * yc * yh * yw;                         \
          if (yc != 1) y_offset += c * yh * yw;                              \
          if (yh != 1) y_offset += h * yw;                                   \
          if (yw != 1) y_offset += w;                                        \
          out_data[x_offset] = MATHOP(out_data[x_offset], y_data[y_offset]); \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  }

template <class T>
T add(T a, T b) {
  return a + b;
}

template <class T>
T sub(T a, T b) {
  return a - b;
}

template <class T>
T mul(T a, T b) {
  return a * b;
}

template <class T>
T div(T a, T b) {
  return a / b;
}

template <class T>
T floordiv(T a, T b) {
  return static_cast<T>(std::trunc(a / b));
}

template <class T>
T pow(T a, T b) {
  return std::pow(a, b);
}

template <class T>
T max(T a, T b) {
  return std::max(a, b);
}

template <class T>
T min(T a, T b) {
  return std::min(a, b);
}

template <class T>
T mod(T a, T b) {
  T res = a % b;
  if ((res != 0) && ((res < 0) != (b < 0))) res += b;
  return res;
}

#ifdef ENABLE_ARM_FP16
template <>
float16_t mod<float16_t>(float16_t a, float16_t b) {
  float16_t res = fmod(a, b);
  if ((res != 0) && ((b < 0) != (res < 0))) res += b;
  return res;
}
#endif

template <>
float mod<float>(float a, float b) {
  float res = fmod(a, b);
  if ((res != 0) && ((b < 0) != (res < 0))) res += b;
  return res;
}

template <class T>
T NaiveTanh(T a) {
  float x = expf(a);
  float y = expf(-a);
  return (x - y) / (x + y);
}

template <class T>
T NaiveSigmoid(T a) {
  const T min = -40.0;  // SIGMOID_THRESHOLD_MIN;
  const T max = 13.0;   // SIGMOID_THRESHOLD_MAX;
  T tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-tmp));
}

template <class T = float>
class ElementwiseComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string y_ = "y";
  std::string out_ = "out";
  // add, sub, mul, div, max
  std::string elt_type_ = "";
  DDim x_dims_{{1, 2, 3, 4}};
  DDim y_dims_{{1, 2, 3, 4}};
  int axis_ = 1;
  std::string act_type_ = "";

 public:
  ElementwiseComputeTester(const Place& place,
                           const std::string& alias,
                           std::string elt_type = "add",
                           std::vector<int64_t> x_shape = {1, 2, 3, 4},
                           std::vector<int64_t> y_shape = {1, 2, 3, 4},
                           int axis = 1,
                           std::string act_type = "")
      : TestCase(place, alias),
        elt_type_(elt_type),
        x_dims_(DDim(x_shape)),
        y_dims_(DDim(y_shape)),
        axis_(axis),
        act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
    if (axis_ < 0) {
      axis_ = x_dims_.size() - y_dims_.size();
    }
    auto x_shape = x_dims_.Vectorize();
    while (x_shape.size() < 4) {
      x_shape.push_back(1);
    }
    auto y_shape = y_dims_.Vectorize();
    y_shape.insert(y_shape.begin(), axis_, 1);
    while (y_shape.size() < 4) {
      y_shape.push_back(1);
    }
    CHECK_EQ(x_shape.size(), 4);
    CHECK_EQ(y_shape.size(), 4);

    auto x = scope->FindTensor(x_);
    auto y = scope->FindTensor(y_);
    auto x_data = x->template data<T>();
    auto y_data = y->template data<T>();
    auto out = scope->NewTensor(out_);
    out->Resize(x_dims_);
    auto out_data = out->template mutable_data<T>();
    memcpy(out_data, x_data, sizeof(T) * x_dims_.production());

    int xn = x_shape[0];
    int xc = x_shape[1];
    int xh = x_shape[2];
    int xw = x_shape[3];

    int yn = y_shape[0];
    int yc = y_shape[1];
    int yh = y_shape[2];
    int yw = y_shape[3];

    if (elt_type_ == "add") {
      ELT(add);
    } else if (elt_type_ == "sub") {
      ELT(sub);
    } else if (elt_type_ == "mul") {
      ELT(mul);
    } else if (elt_type_ == "div") {
      ELT(div);
    } else if (elt_type_ == "floordiv") {
      ELT(floordiv);
    } else if (elt_type_ == "max") {
      ELT(max);
    } else if (elt_type_ == "min") {
      ELT(min);
    } else if (elt_type_ == "pow") {
      ELT(pow);
    } else if (elt_type_ == "mod") {
      ELT(mod);
    } else {
      LOG(FATAL) << "unsupported op";
    }
#ifdef LITE_WITH_X86
    if (!act_type_.empty()) {
      if (act_type_ == "relu") {
        for (int i = 0; i < x_dims_.production(); i++) {
          out_data[i] = std::max(static_cast<T>(0), out_data[i]);
        }
      } else if (act_type_ == "tanh") {
        for (int i = 0; i < x_dims_.production(); i++)
          out_data[i] = NaiveTanh(out_data[i]);
      } else if (act_type_ == "sigmoid") {
        for (int i = 0; i < x_dims_.production(); i++)
          out_data[i] = NaiveSigmoid(out_data[i]);
      } else {
        LOG(FATAL) << "unsupported act_type:" << act_type_;
      }
    }
#else
    if (!act_type_.empty()) {
      if (act_type_ == "relu") {
        for (int i = 0; i < x_dims_.production(); i++) {
          out_data[i] = std::max(static_cast<T>(0), out_data[i]);
        }
      } else {
        LOG(FATAL) << "unsupported act_type:" << act_type_;
      }
    }
#endif
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    std::string op_type = "elementwise_" + elt_type_;
    if (!act_type_.empty()) {
      op_type = "fusion_" + op_type + "_activation";
    }
    op_desc->SetType(op_type);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axis", axis_);
    if (!act_type_.empty()) {
      op_desc->SetAttr("act_type", act_type_);
    }
  }

  void PrepareData() override {
    std::vector<T> dx(x_dims_.production());
    for (size_t i = 0; i < dx.size(); i++) {
      dx[i] = static_cast<T>((i % 3) * 1.1f);
      dx[i] = static_cast<T>(dx[i] == static_cast<T>(0) ? 1 : dx[i]);
    }
    SetCommonTensor(x_, x_dims_, dx.data());

    std::vector<T> dy(y_dims_.production());
    for (size_t i = 0; i < dy.size(); i++) {
      dy[i] = static_cast<T>((i % 5) * 1.1f);
      dy[i] = static_cast<T>(dy[i] == static_cast<T>(0) ? 1 : dy[i]);
    }
    SetCommonTensor(y_, y_dims_, dy.data());
  }
};

// add sub mul div max   +act
template <class T = float>
void TestElt(Place place,
             float abs_error,
             std::string elt_type,
             std::vector<int64_t> x_shape,
             std::vector<int64_t> y_shape,
             int axis,
             std::string act_type = "",
             std::string func_type = "def") {
#if defined(LITE_WITH_XPU)
  if ((y_shape.size() != 1 && x_shape.size() != y_shape.size()) ||
      elt_type != std::string("add") || !act_type.empty()) {
    return;
  }
#endif
#if defined(NNADAPTER_WITH_CAMBRICON_MLU)
  if (elt_type == std::string("max") || elt_type == std::string("min") ||
      x_shape.size() != y_shape.size()) {
    return;
  }
#endif
  std::unique_ptr<arena::TestCase> tester(new ElementwiseComputeTester<T>(
      place, func_type, elt_type, x_shape, y_shape, axis, act_type));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestEltDims(Place place, float abs_error) {
  TestElt(place, abs_error, "add", {2, 3, 4, 5}, {2, 3, 4, 5}, 0);
  TestElt(place, abs_error, "add", {2, 3, 4}, {2, 3, 4}, 0);
  TestElt(place, abs_error, "add", {2, 3, 4}, {2, 3}, 0);
  TestElt(place, abs_error, "add", {2, 3}, {2}, 0);
  TestElt(place, abs_error, "add", {2, 3, 4, 5}, {3, 4}, 1);
  TestElt(place, abs_error, "add", {2, 3, 4}, {3, 4}, 1);
  TestElt(place, abs_error, "add", {2, 3}, {3}, 1);
  TestElt(place, abs_error, "add", {2, 3, 4, 5}, {4, 5}, 2);
  TestElt(place, abs_error, "add", {2, 3, 4}, {4}, 2);
  TestElt(place, abs_error, "add", {2, 3, 4, 5}, {5}, 3);
  TestElt(place, abs_error, "add", {2, 3, 4, 5}, {3, 4, 5}, -1);
  TestElt(place, abs_error, "add", {2, 3, 4}, {3, 4}, -1);
}

void TestEltTypes(Place place, float abs_error) {
  for (auto elt_type : std::vector<std::string>{
           "add", "sub", "mul", "div", "max", "min", "pow"}) {
    TestElt(place, abs_error, elt_type, {2, 3, 4, 5}, {2, 3, 4, 5}, 0);
    TestElt(place, abs_error, elt_type, {2, 3, 4, 5}, {3}, 1);
  }

  if (place.target == TARGET(kARM)) {
    Place arm_int32_place(TARGET(kARM), PRECISION(kInt32));
    TestElt<int32_t>(
        arm_int32_place, abs_error, "floordiv", {2, 3, 4, 5}, {2, 3, 4, 5}, 0);
    Place arm_int64_place(TARGET(kARM), PRECISION(kInt64));
    TestElt<int64_t>(
        arm_int64_place, abs_error, "floordiv", {2, 3, 4, 5}, {3}, 1);
  }

  if (place.target == TARGET(kX86)) {
    Place x86place(TARGET(kX86));
    for (auto op : std::vector<std::string>{
             "add", "sub", "mul", "div", "floordiv", "max", "min", "pow"}) {
      TestElt<float>(x86place, abs_error, op, {2, 3, 4, 15}, {2, 3, 4, 15}, 0);
      TestElt<int>(x86place,
                   abs_error,
                   op,
                   {2, 3, 14, 5},
                   {2, 3, 14, 5},
                   0,
                   "",
                   "int32");
      TestElt<int64_t>(x86place,
                       abs_error,
                       op,
                       {2, 13, 4, 5},
                       {2, 13, 4, 5},
                       0,
                       "",
                       "int64");
    }
    TestElt<int>(x86place,
                 abs_error,
                 "mod",
                 {12, 3, 4, 5},
                 {12, 3, 4, 5},
                 0,
                 "",
                 "int32");
    TestElt<int64_t>(
        x86place, abs_error, "mod", {2, 3, 4, 5}, {2, 3, 4, 5}, 0, "", "int64");
  }
}

void TestEltFuseAct(Place place, float abs_error) {
  for (auto elt_type :
       std::vector<std::string>{"add", "sub", "mul", "div", "min", "max"}) {
    TestElt(place, abs_error, elt_type, {2, 3, 4, 5}, {2, 3, 4, 5}, 0, "relu");
    TestElt(place, abs_error, elt_type, {2, 3, 4, 5}, {3}, 1, "relu");
  }
}

#ifdef LITE_WITH_X86
void TestEltFuseActFloat(Place place, float abs_error) {
  for (auto elt_type :
       std::vector<std::string>{"add", "sub", "mul", "div", "min", "max"}) {
    TestElt<float>(
        place, abs_error, elt_type, {2, 13, 4, 5}, {2, 13, 4, 5}, 0, "relu");
    TestElt<float>(
        place, abs_error, elt_type, {2, 13, 4, 5}, {2, 13, 4, 5}, 0, "tanh");
    TestElt<float>(
        place, abs_error, elt_type, {2, 13, 4, 5}, {2, 13, 4, 5}, 0, "sigmoid");
    TestElt<float>(place, abs_error, elt_type, {2, 3, 14, 5}, {3}, 1, "relu");
    TestElt<float>(place, abs_error, elt_type, {2, 3, 14, 5}, {3}, 1, "tanh");
    TestElt<float>(
        place, abs_error, elt_type, {2, 3, 14, 5}, {3}, 1, "sigmoid");
  }
}
#endif

#ifdef ENABLE_ARM_FP16
void TestFp16EltDims(Place place, float abs_error, std::string test_operator) {
  TestElt<float16_t>(
      place, abs_error, test_operator, {1, 40, 40, 40}, {1, 40, 40, 40}, 0);
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {2, 3, 4, 5}, 0);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4}, {2, 3, 4}, 0);
  TestElt<float16_t>(
      place, abs_error, test_operator, {1, 40, 40, 40}, {1, 40, 1, 1}, 0);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4}, {2, 3}, 0);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3}, {2}, 0);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4, 5}, {3, 4}, 1);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4}, {3, 4}, 1);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3}, {3}, 1);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4, 5}, {4, 5}, 2);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4}, {4}, 2);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4, 5}, {5}, 3);
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {3, 4, 5}, -1);
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3, 4}, {3, 4}, -1);
}

void TestFp16EltFuseAct(Place place,
                        float abs_error,
                        std::string test_operator) {
  TestElt<float16_t>(place,
                     abs_error,
                     test_operator,
                     {1, 40, 40, 40},
                     {1, 40, 1, 1},
                     0,
                     "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {2, 3, 4, 5}, 0, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4}, {2, 3, 4}, 0, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4}, {2, 3}, 0, "relu");
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3}, {2}, 0, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {3, 4}, 1, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4}, {3, 4}, 1, "relu");
  TestElt<float16_t>(place, abs_error, test_operator, {2, 3}, {3}, 1, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {4, 5}, 2, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4}, {4}, 2, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {5}, 3, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4, 5}, {3, 4, 5}, -1, "relu");
  TestElt<float16_t>(
      place, abs_error, test_operator, {2, 3, 4}, {3, 4}, -1, "relu");
}
#endif

TEST(Elementwise, precision) {
  Place place;
  float abs_error = 2e-5;

#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-1;
  for (auto elt_type : std::vector<std::string>{
           "add", "sub", "mul", "div", "max", "min", "pow"}) {
    TestElt(place, abs_error, elt_type, {2, 3, 4, 5}, {2, 3, 4, 5}, 0);
  }
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif
#ifdef ENABLE_ARM_FP16
  Place place1(TARGET(kARM), PRECISION(kFP16));
  TestFp16EltDims(place1, abs_error, "add");
  TestFp16EltFuseAct(place1, abs_error, "add");
  TestFp16EltDims(place1, abs_error, "mul");
  TestFp16EltFuseAct(place1, abs_error, "mul");
  TestFp16EltDims(place1, abs_error, "sub");
  TestFp16EltFuseAct(place1, abs_error, "sub");
  TestFp16EltDims(place1, abs_error, "div");
  TestFp16EltFuseAct(place1, abs_error, "div");
#endif

  TestEltDims(place, abs_error);
  TestEltTypes(place, abs_error);
  TestEltFuseAct(place, abs_error);
#ifdef LITE_WITH_X86
  TestEltFuseActFloat(place, abs_error);
#endif
}

}  // namespace lite
}  // namespace paddle
