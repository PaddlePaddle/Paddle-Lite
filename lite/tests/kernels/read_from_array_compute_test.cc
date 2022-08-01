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

class ReadFromArrayComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string idn_ = "i";
  std::string out_ = "out";
  DDim tar_dims_{{3, 5, 4, 4}};
  int x_size_ = 1;
  int id_ = 0;

 public:
  ReadFromArrayComputeTester(const Place& place,
                             const std::string& alias,
                             DDim tar_dims,
                             int x_size = 1,
                             int id = 0)
      : TestCase(place, alias), tar_dims_(tar_dims), x_size_(x_size), id_(id) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindVar(x_)->GetMutable<std::vector<Tensor>>();
    auto idn = scope->FindTensor(idn_);
    auto out = scope->NewTensor(out_);

    int id = idn->data<int64_t>()[0];
    out->CopyDataFrom(x->at(id));
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("read_from_array");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("I", {idn_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<DDim> x_dims(x_size_);
    std::vector<std::vector<float>> x_data(x_size_);
    for (int i = 0; i < x_size_; i++) {
      x_dims[i] = tar_dims_;
      x_data[i].resize(x_dims[i].production());
      fill_data_rand(x_data[i].data(), -1.f, 1.f, x_dims[i].production());
    }
    SetCommonTensorList(x_, x_dims, x_data);

    std::vector<int64_t> didn(1);
    didn[0] = id_;
    SetCommonTensor(idn_, DDim{{1}}, didn.data());
  }
};

void TestReadFromArray(Place place, float abs_error) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int x_size : {1, 3}) {
    for (int id : {0, 2}) {
      if (x_size < id + 1) continue;
      std::unique_ptr<arena::TestCase> tester(
          new ReadFromArrayComputeTester(place, "def", dims, x_size, id));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(ReadFromArray, precision) {
  Place place;
  float abs_error = 1e-5;
#if 0 && defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestReadFromArray(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
