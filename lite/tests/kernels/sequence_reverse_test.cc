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
#include <cstring>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <class T>
class SequenceReverseTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string y_ = "y";
  DDim x_dims_{{9, 2, 3, 4}};
  LoD x_lod_{{{0, 2, 5, 9}}};

 public:
  SequenceReverseTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* y = scope->NewTensor(y_);
    y->Resize(x_dims_);
    y->set_lod(x_lod_);
    auto* x = scope->FindTensor(x_);
    auto* x_data = x->template data<T>();
    auto* y_data = y->template mutable_data<T>();

    const auto lod = x_lod_.back();
    size_t row_numel = static_cast<size_t>(x_dims_.production() / x_dims_[0]);

    for (size_t idx = 0; idx < lod.size() - 1; idx++) {
      size_t start_pos = lod[idx];
      size_t end_pos = lod[idx + 1];
      for (auto pos = start_pos; pos < end_pos; pos++) {
        size_t cur_pos = end_pos - pos - 1 + start_pos;
        std::memcpy(y_data + pos * row_numel,
                    x_data + cur_pos * row_numel,
                    row_numel * sizeof(T));
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_reverse");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Y", {y_});
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand<T>(x_data.data(), -10, 10, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data(), x_lod_);
  }
};

template <class T>
void TestSequenceReverse(const Place place,
                         const float abs_error,
                         const std::string alias) {
  std::unique_ptr<arena::TestCase> tester(
      new SequenceReverseTester<T>(place, alias));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(sequence_reverse, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  TestSequenceReverse<float>(place, abs_error, "def");
  TestSequenceReverse<int>(place, abs_error, "int32");
  TestSequenceReverse<int64_t>(place, abs_error, "int64");
}

}  // namespace lite
}  // namespace paddle
