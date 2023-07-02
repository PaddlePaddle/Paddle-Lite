// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

#define BITWISE(MATHOP)                                                      \
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
T and_op(T a, T b) {
  return a & b;
}

template <>
bool and_op<bool>(bool a, bool b) {
  return a && b;
}

template <class T>
T or_op(T a, T b) {
  return a | b;
}

template <>
bool or_op<bool>(bool a, bool b) {
  return a || b;
}

template <class T>
T xor_op(T a, T b) {
  return a ^ b;
}

template <>
bool xor_op<bool>(bool a, bool b) {
  return a != b;
}

template <class T>
T not_op(T a) {
  return ~a;
}

template <>
bool not_op(bool a) {
  return !a;
}

template <class T = int64_t>
class BitwiseComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string y_ = "Y";
  std::string out_ = "Out";
  std::string bitwise_type_ = "and";  // ["and", "or", "xor", "not"]
  DDim x_dims_;
  DDim y_dims_;
  int axis_ = -1;

 public:
  BitwiseComputeTester(const Place& place,
                       const std::string& alias,
                       std::vector<int64_t> x_shape,
                       std::vector<int64_t> y_shape,
                       std::string bitwise_type = "and",
                       int axis = -1)
      : TestCase(place, alias),
        bitwise_type_(bitwise_type),
        x_dims_(DDim(x_shape)),
        y_dims_(DDim(y_shape)),
        axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto x_data = x->template data<T>();
    auto out = scope->NewTensor(out_);
    out->Resize(x_dims_);
    auto out_data = out->template mutable_data<T>();
    memcpy(out_data, x_data, sizeof(T) * x_dims_.production());

    if (bitwise_type_ == "not") {
      auto numel = x->numel();
      for (int i = 0; i < numel; ++i) {
        out_data[i] = not_op(x_data[i]);
      }
    } else {
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

      auto y = scope->FindTensor(y_);
      auto y_data = y->template data<T>();

      int xn = x_shape[0];
      int xc = x_shape[1];
      int xh = x_shape[2];
      int xw = x_shape[3];

      int yn = y_shape[0];
      int yc = y_shape[1];
      int yh = y_shape[2];
      int yw = y_shape[3];
      if (bitwise_type_ == "and") {
        BITWISE(and_op);
      } else if (bitwise_type_ == "or") {
        BITWISE(or_op);
      } else if (bitwise_type_ == "xor") {
        BITWISE(xor_op);
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    std::string op_type = "bitwise_" + bitwise_type_;
    op_desc->SetType(op_type);
    op_desc->SetInput("X", {x_});
    if (bitwise_type_ != "not") {
      op_desc->SetInput("Y", {y_});
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axis_", axis_);
  }

  void PrepareData();
};

template <class T>
void BitwiseComputeTester<T>::PrepareData() {
  std::vector<T> dx(x_dims_.production());
  for (int i = 0; i < x_dims_.production(); i++) {
    dx[i] = static_cast<T>(i % 128);
  }
  SetCommonTensor(x_, x_dims_, dx.data());
  if (bitwise_type_ == "not") {
    return;
  } else {
    std::vector<T> dy(y_dims_.production());
    for (int i = 0; i < y_dims_.production(); i++) {
      dy[i] = static_cast<T>(i % 128);
    }
    SetCommonTensor(y_, y_dims_, dy.data());
  }
}

template <>
void BitwiseComputeTester<bool>::PrepareData() {
  std::vector<uint8_t> x_data(x_dims_.production());
  for (int i = 0; i < x_dims_.production(); i++) {
    x_data[i] = static_cast<uint8_t>(i % 2);
  }
  SetCommonTensor(x_, x_dims_, reinterpret_cast<bool*>(x_data.data()));
  if (bitwise_type_ == "not") {
    return;
  } else {
    std::vector<uint8_t> y_data(y_dims_.production());
    for (int i = 0; i < y_dims_.production(); i++) {
      y_data[i] = static_cast<uint8_t>(i % 2);
    }
    SetCommonTensor(y_, y_dims_, reinterpret_cast<bool*>(y_data.data()));
  }
}

template <class T = int64_t>
void TestPre(const Place& place,
             float abs_error,
             std::vector<int64_t> x_shape,
             std::vector<int64_t> y_shape,
             std::string bitwise_type,
             const std::string& alias = "int64") {
  std::unique_ptr<arena::TestCase> tester(new BitwiseComputeTester<T>(
      place, alias, x_shape, y_shape, bitwise_type));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = bool>
void TestPer(const Place& place,
             float abs_error,
             std::vector<int64_t> x_shape,
             std::vector<int64_t> y_shape,
             std::string bitwise_type,
             const std::string& alias = "int64") {
  std::unique_ptr<arena::TestCase> tester(new BitwiseComputeTester<T>(
      place, alias, x_shape, y_shape, bitwise_type));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPerformance();
}

void TestBitDims(const Place& place, float abs_error) {
  for (auto bit_type : std::vector<std::string>{"and", "not"}) {
    if (bit_type != "not") {
      TestPre(place, abs_error, {2, 3, 4, 5}, {3, 4, 5}, bit_type);
      TestPre(place, abs_error, {2, 3, 4}, {3, 4}, bit_type);
      TestPre(place, abs_error, {2, 3}, {2, 1}, bit_type);
    } else {
      TestPre(place, abs_error, {2}, {}, bit_type);
    }
  }
}

void TestBitTypes(const Place& place, float abs_error) {
  Place host_bool_place(TARGET(kHost), PRECISION(kBool));
  Place host_int32_place(TARGET(kHost), PRECISION(kInt32));
  Place host_int64_place(TARGET(kHost), PRECISION(kInt64));
  for (auto bit_type : std::vector<std::string>{"and", "not"}) {
    if (bit_type == "not") {
      TestPre<bool>(
          host_bool_place, abs_error, {2, 3, 4, 5}, {}, bit_type, "bl");
      TestPre<int32_t>(
          host_int32_place, abs_error, {2, 3, 4, 5}, {}, bit_type, "int32");
      TestPre<int64_t>(
          host_int64_place, abs_error, {2, 3, 4, 5}, {}, bit_type, "int64");
    } else {
      TestPre<bool>(host_bool_place,
                    abs_error,
                    {2, 3, 4, 5},
                    {2, 3, 4, 5},
                    bit_type,
                    "bl");
      TestPre<int32_t>(host_int32_place,
                       abs_error,
                       {2, 3, 4, 5},
                       {3, 4, 5},
                       bit_type,
                       "int32");
      TestPre<int64_t>(
          host_int64_place, abs_error, {2, 3, 4, 5}, {4, 1}, bit_type, "int64");
    }
  }
}

TEST(BT, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  TestBitDims(place, abs_error);
  TestBitTypes(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
