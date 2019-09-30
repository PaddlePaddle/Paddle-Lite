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

namespace paddle {
namespace lite {

class RangeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string start = "Start";
  std::string end = "End";
  std::string step = "Step";
  std::string out = "Out";
  int st_, ed_, sp_;

 public:
  RangeComputeTester(const Place& place,
                     const std::string& alias,
                     float st,
                     float ed,
                     float sp)
      : TestCase(place, alias), st_(st), ed_(ed), sp_(sp) {}

  void RunBaseline(Scope* scope) override {
    auto* output = scope->NewTensor(out);
    CHECK(output);
    int64_t size;
    auto* st = scope->FindMutableTensor(start);
    auto* ed = scope->FindMutableTensor(end);
    auto* sp = scope->FindMutableTensor(step);
    float st_val = st->data<float>()[0];
    float ed_val = ed->data<float>()[0];
    float sp_val = sp->data<float>()[0];
    // size = (std::abs(ed_val - st_val) + std::abs(sp_val) - 1) /
    // std::abs(sp_val);
    size = std::ceil(std::abs((ed_val - st_val) / sp_val));
    output->Resize(DDim(std::vector<int64_t>({static_cast<int>(size)})));
    auto* out_data = output->mutable_data<float>();

    float val = st_;
    for (int i = 0; i < size; i++) {
      out_data[i] = val;
      val += sp_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("range");
    op_desc->SetInput("Start", {start});
    op_desc->SetInput("End", {end});
    op_desc->SetInput("Step", {step});
    op_desc->SetOutput("Out", {out});
  }

  void PrepareData() override {
    std::vector<float> st(1);
    std::vector<float> ed(1);
    std::vector<float> sp(1);

    st[0] = st_;
    ed[0] = ed_;
    sp[0] = sp_;
    DDim dim(std::vector<int64_t>({1}));

    SetCommonTensor(start, dim, st.data());
    SetCommonTensor(end, dim, ed.data());
    SetCommonTensor(step, dim, sp.data());
  }
};

void test_range(Place place) {
  std::unique_ptr<arena::TestCase> tester1(
      new RangeComputeTester(place, "def", 1, 10, 1));
  arena::Arena arena(std::move(tester1), place, 2e-5);
  arena.TestPrecision();

  std::unique_ptr<arena::TestCase> tester2(
      new RangeComputeTester(place, "def", 10, 1, -2));
  arena::Arena arena2(std::move(tester2), place, 2e-5);
  arena2.TestPrecision();
}

TEST(Range, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_range(place);
#endif
}

}  // namespace lite
}  // namespace paddle
