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

namespace paddle {
namespace lite {

template <class T>
class RangeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string start = "Start";
  std::string end = "End";
  std::string step = "Step";
  std::string out = "Out";
  T st_, ed_, sp_;

 public:
  RangeComputeTester(
      const Place& place, const std::string& alias, T st, T ed, T sp)
      : TestCase(place, alias), st_(st), ed_(ed), sp_(sp) {}

  void RunBaseline(Scope* scope) override {
    auto* output = scope->NewTensor(out);
    CHECK(output);

    auto* st = scope->FindTensor(start);
    auto* ed = scope->FindTensor(end);
    auto* sp = scope->FindTensor(step);
    T st_val = st->template data<T>()[0];
    T ed_val = ed->template data<T>()[0];
    T sp_val = sp->template data<T>()[0];
    int64_t size = std::is_integral<T>::value
                       ? ((std::abs(ed_val - st_val) + std::abs(sp_val) - 1) /
                          std::abs(sp_val))
                       : std::ceil(std::abs((ed_val - st_val) / sp_val));
    output->Resize(DDim(std::vector<int64_t>({static_cast<int>(size)})));
    auto* out_data = output->template mutable_data<T>();

    T val = st_;
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
    std::vector<T> st(1);
    std::vector<T> ed(1);
    std::vector<T> sp(1);

    st[0] = st_;
    ed[0] = ed_;
    sp[0] = sp_;
    DDim dim(std::vector<int64_t>({1}));

    SetCommonTensor(start, dim, st.data(), {}, true);
    SetCommonTensor(end, dim, ed.data(), {}, true);
    SetCommonTensor(step, dim, sp.data(), {}, true);
  }
};

template <class T>
void test_range(Place place, float abs_error) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();

  std::unique_ptr<arena::TestCase> tester1(
      new RangeComputeTester<T>(place, "def", 1, 10, 1));
  arena::Arena arena(std::move(tester1), place, abs_error);
  arena.TestPrecision();

  std::unique_ptr<arena::TestCase> tester2(
      new RangeComputeTester<T>(place, "def", 10, 1, -2));
  arena::Arena arena2(std::move(tester2), place, abs_error);
  arena2.TestPrecision();
}

TEST(Range, precision) {
  Place place;
  float abs_error = 1e-5;

#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  test_range<int64_t>(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  test_range<float>(place, abs_error);
  test_range<int>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
