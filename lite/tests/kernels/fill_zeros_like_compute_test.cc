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
#include <cstring>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <class T>
class FillZerosLikeComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string out_ = "out";
  DDim x_dims_;

 public:
  FillZerosLikeComputeTester(const Place& place,
                             const std::string& alias,
                             const DDim& x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* x = scope->FindTensor(x_);
    out->Resize(x_dims_);
    out->set_lod(x->lod());
    auto* out_data = out->template mutable_data<T>();
    for (int64_t i = 0; i < out->numel(); i++) {
      out_data[i] = static_cast<T>(0);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fill_zeros_like");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<T> x(x_dims_.production());
    fill_data_rand(
        x.data(), static_cast<T>(-1), static_cast<T>(1), x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());
  }
};

template <class T>
void TestFillZerosLike(Place place, float abs_error) {
  std::vector<std::vector<int64_t>> x_shapes{
      {2, 3, 4, 5}, {2, 3, 4}, {3, 4}, {4}};
  std::string alias("def");
  auto precision = lite_api::PrecisionTypeTrait<T>::Type();
  switch (precision) {
    case PRECISION(kFloat):
      alias = std::string("float32");
      break;
    case PRECISION(kInt32):
      alias = std::string("int32");
      break;
    case PRECISION(kInt64):
      alias = std::string("int64");
      break;
    default:
      LOG(FATAL) << "unsupported data type: "
                 << lite_api::PrecisionToStr(precision);
  }
  for (auto x_shape : x_shapes) {
    std::unique_ptr<arena::TestCase> tester(
        new FillZerosLikeComputeTester<T>(place, alias, DDim(x_shape)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(fill_zeros_like, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestFillZerosLike<float>(place, abs_error);
  TestFillZerosLike<int>(place, abs_error);
  TestFillZerosLike<int64_t>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
