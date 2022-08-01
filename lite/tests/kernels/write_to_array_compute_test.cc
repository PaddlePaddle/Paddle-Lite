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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class WriteToArrayComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string idn_ = "i";
  std::string out_ = "out";
  DDim x_dims_{{3, 5, 4, 4}};
  int out_size_ = 0;
  int id_ = 0;

 public:
  WriteToArrayComputeTester(const Place& place,
                            const std::string& alias,
                            DDim x_dims,
                            int out_size = 0,
                            int id = 0)
      : TestCase(place, alias), x_dims_(x_dims), out_size_(out_size), id_(id) {}

  void RunBaseline(Scope* scope) override {
    auto out = scope->Var(out_)->GetMutable<std::vector<Tensor>>();
    auto x = scope->FindTensor(x_);

    if (out->size() < id_ + 1) {
      out->resize(id_ + 1);
    }
    out->at(id_).Resize(x->dims());
    auto out_data = out->at(id_).mutable_data<float>();
    memcpy(out_data, x->data<float>(), sizeof(float) * x->numel());
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("write_to_array");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("I", {idn_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> dx(x_dims_.production());
    fill_data_rand(dx.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());

    std::vector<int64_t> didn(1);
    didn[0] = id_;
    SetCommonTensor(idn_, DDim{{1}}, didn.data());
  }
};

void TestWriteToArray(Place place, float abs_error) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int out_size : {0, 3}) {
    for (int id : {0, 1, 4}) {
      std::unique_ptr<arena::TestCase> tester(
          new WriteToArrayComputeTester(place, "def", dims, out_size, id));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(WriteToArray, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestWriteToArray(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
