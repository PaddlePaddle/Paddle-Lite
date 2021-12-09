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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

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
class SumComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input1_ = "X1";
  std::string input2_ = "X2";
  std::string input3_ = "X3";
  std::string output_ = "Out";
  DDim dims_{{1, 5, 6, 7}};

 public:
  SumComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    std::vector<const lite::Tensor*> x;
    x.emplace_back(scope->FindTensor(input1_));
    x.emplace_back(scope->FindTensor(input2_));
    x.emplace_back(scope->FindTensor(input3_));
    auto x_dims = x[0]->dims();
    int rank = x_dims.size();
    CHECK_EQ(rank, 4);
    auto x_shape = x_dims.Vectorize();
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(x_dims);
    auto out_data = out->template mutable_data<T>();
    memcpy(out_data, x[0]->template data<T>(), sizeof(T) * x_dims.production());

    int xn = x_shape[0];
    int xc = x_shape[1];
    int xh = x_shape[2];
    int xw = x_shape[3];

    int yn = xn;
    int yc = xc;
    int yh = xh;
    int yw = xw;

    for (auto it = x.begin() + 1; it != x.end(); ++it) {
      auto y_data = (*it)->data<T>();
      ELT(add);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sum");
    op_desc->SetInput("X", {input1_, input2_, input3_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<T> data(dims_.production());
    fill_data_rand(data.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor<T>(input1_, dims_, data.data());
    SetCommonTensor<T>(input2_, dims_, data.data());
    SetCommonTensor<T>(input3_, dims_, data.data());
  }
};

template <class T = float>
void test_sum(Place place, float abs_error) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::unique_ptr<arena::TestCase> tester(
      new SumComputeTester<T>(place, "def"));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Sum, precision) {
  Place place;
  float abs_error = 2e-4;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#else
  return;
#endif
#else
  return;
#endif

  test_sum<float>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
