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

class LoDArrayLengthComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  DDim tar_dims_{{3, 5, 4, 4}};

 public:
  LoDArrayLengthComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    const auto* x = scope->FindVar(x_)->GetMutable<std::vector<Tensor>>();
    auto* out = scope->NewTensor(out_);

    out->Resize(DDim({1}));
    auto* out_data = out->mutable_data<int64_t>();
    out_data[0] = static_cast<int64_t>(x->size());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("lod_array_length");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    constexpr int x_size_ = 10;
    std::vector<DDim> x_dims(x_size_);
    std::vector<std::vector<float>> x_data(x_size_);
    for (int i = 0; i < x_size_; i++) {
      x_dims[i] = tar_dims_;
      x_data[i].resize(x_dims[i].production());
      fill_data_rand(x_data[i].data(), -1.f, 1.f, x_dims[i].production());
    }
    SetCommonTensorList(x_, x_dims, x_data);
  }
};

void LoDArrayLengthTestHelper(Place place, float abs_error) {
  std::unique_ptr<arena::TestCase> tester(
      new LoDArrayLengthComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestLoDArrayLength(Place place, float abs_error) {
  LoDArrayLengthTestHelper(place, abs_error);
}

TEST(lod_array_length, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestLoDArrayLength(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
