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

template <class T>
class TrilTriuComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "X";
  std::string out_ = "Out";
  DDim x_dims_;
  int diagonal_{0};
  bool lower_{true};

 public:
  TrilTriuComputeTester(const Place& place,
                        const std::string& alias,
                        const DDim& x_dims,
                        const int diagonal = 0,
                        const bool lower = true)
      : TestCase(place, alias),
        x_dims_(x_dims),
        diagonal_(diagonal),
        lower_(lower) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* x = scope->FindTensor(x_);
    out->Resize(x_dims_);
    out->set_lod(x->lod());

    auto* x_data = x->template data<T>();
    auto* out_data = out->template mutable_data<T>();
    auto h = x_dims_[x_dims_.size() - 2];
    auto w = x_dims_[x_dims_.size() - 1];
    auto n = x_dims_.production() / h / w;

    for (int64_t i = 0; i < n; i++) {
      for (int64_t idx = 0; idx < h * w; idx++) {
        auto row = idx / w;
        auto col = idx % w;
        bool mask = lower_ ? (col - row > diagonal_) : (col - row < diagonal_);
        out_data[idx] = mask ? 0 : x_data[idx];
      }
      x_data += h * w;
      out_data += h * w;
    }
    return;
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("tril_triu");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("diagonal", diagonal_);
    op_desc->SetAttr("lower", lower_);
    return;
  }

  void PrepareData() override {
    std::vector<T> din(x_dims_.production());
    fill_data_rand(din.data(),
                   static_cast<T>(-10),
                   static_cast<T>(10),
                   x_dims_.production());
    SetCommonTensor(x_, x_dims_, din.data());
    return;
  }
};

template <class T = float>
void TestTrilTriuHelper(Place place,
                        float abs_error,
                        const std::vector<int64_t> x_dims,
                        const int diagonal = 0,
                        const bool lower = true) {
  auto precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::string alias("def");
  switch (precision) {
    case lite_api::PrecisionType::kFloat:
      alias = std::string("float32");
      break;
    default:
      LOG(FATAL) << "unsupported precision: "
                 << lite_api::PrecisionToStr(precision);
  }

  std::unique_ptr<arena::TestCase> tester(new TrilTriuComputeTester<T>(
      place, alias, DDim(x_dims), diagonal, lower));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(cumsum, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  for (auto x_shape :
       std::vector<std::vector<int64_t>>{{3, 4}, {5, 6, 7}, {5, 6, 7, 8}}) {
    for (auto lower : {true, false}) {
      for (auto diagonal : {-1, 0, 2}) {
        TestTrilTriuHelper<float>(place, abs_error, x_shape, diagonal, lower);
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
