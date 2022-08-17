// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace lite {

template <class T>
class CumsumComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "X";
  std::string out_ = "Out";
  DDim x_dims_;
  int axis_{-1};
  bool flatten_{false};
  bool exclusive_{false};
  bool reverse_{false};

 public:
  CumsumComputeTester(const Place& place,
                      const std::string& alias,
                      const DDim& x_dims,
                      const int axis = -1,
                      const bool flatten = false,
                      const bool exclusive = false,
                      const bool reverse = false)
      : TestCase(place, alias),
        x_dims_(x_dims),
        axis_(axis),
        flatten_(flatten),
        exclusive_(exclusive),
        reverse_(reverse) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* x = scope->FindTensor(x_);
    if (flatten_) {
      out->Resize(DDim({x->numel()}));
    } else {
      out->Resize(x->dims());
    }

    auto x_dims = x->dims();
    auto* x_data = x->template data<T>();
    auto* out_data = out->template mutable_data<T>();

    if (flatten_ || x_dims.size() == 1) {
      int64_t x_size = x->numel();
      out_data[0] = x_data[0];
      for (int64_t i = 1; i < x_size; i++) {
        out_data[i] = x_data[i] + out_data[i - 1];
      }
    } else {
      int axis = axis_ < 0 ? axis_ + x_dims.size() : axis_;
      int64_t pre = x_dims.count(0, axis);
      int64_t count = x_dims[axis];
      int64_t post = x_dims.count(axis + 1, x_dims.size());

      for (int64_t i = 0; i < pre; i++) {
        for (int64_t j = 0; j < post; j++) {
          int64_t step = i * count * post + j;
          const T* src = x_data + step;
          T* dst = out_data + step;
          dst[0] = src[0];
          for (int64_t k = 1; k < count; k++) {
            dst[k * post] = src[k * post] + dst[(k - 1) * post];
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("cumsum");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("flatten", flatten_);
    op_desc->SetAttr("exclusive", exclusive_);
    op_desc->SetAttr("reverse", reverse_);
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    for (int64_t i = 0; i < x_dims_.production(); i++) {
      x_data[i] = i;
    }
    SetCommonTensor(x_, x_dims_, x_data.data());
  }
};

template <class T = float>
void TestCumsumHelper(Place place,
                      float abs_error,
                      const std::vector<int64_t> x_dims,
                      const int axis = -1,
                      const bool flatten = false,
                      const bool exclusive = false,
                      const bool reverse = false,
                      const std::string& alias = "def") {
  std::unique_ptr<arena::TestCase> tester(new CumsumComputeTester<T>(
      place, alias, DDim(x_dims), axis, flatten, exclusive, reverse));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestCumsumAxis(Place place,
                    float abs_error,
                    const std::string& alias = "def") {
  std::vector<std::vector<int64_t>> shapes{
      {10}, {4, 5}, {3, 4, 5}, {2, 3, 4, 5}};
  std::vector<int> axes{-4, -3, -2, -1, 0, 1, 2, 3};

  for (auto x_shape : shapes) {
    for (auto axis : axes) {
      if (axis < (-1) * static_cast<int>(x_shape.size()) ||
          axis >= static_cast<int>(x_shape.size()))
        continue;
      TestCumsumHelper<T>(
          place, abs_error, x_shape, axis, false, false, false, alias);
    }
  }
}

template <class T = float>
void TestCumsumFlatten(Place place,
                       float abs_error,
                       const std::string& alias = "def") {
  std::vector<std::vector<int64_t>> shapes{{10}, {2, 3, 4, 5}};
  for (auto x_shape : shapes) {
    TestCumsumHelper<T>(
        place, abs_error, x_shape, -1, true, false, false, alias);
  }
}

TEST(cumsum, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  TestCumsumAxis<float>(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
  TestCumsumAxis<float>(place, abs_error);
  TestCumsumAxis<int32_t>(place, abs_error);
  TestCumsumAxis<int64_t>(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  TestCumsumAxis<int32_t>(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestCumsumAxis<float>(place, abs_error, "float32");
  TestCumsumFlatten<float>(place, abs_error, "float32");

  TestCumsumAxis<int32_t>(place, abs_error, "int32");
  TestCumsumFlatten<int32_t>(place, abs_error, "int32");

  TestCumsumAxis<int64_t>(place, abs_error, "int64");
  TestCumsumFlatten<int64_t>(place, abs_error, "int64");
}

}  // namespace lite
}  // namespace paddle
