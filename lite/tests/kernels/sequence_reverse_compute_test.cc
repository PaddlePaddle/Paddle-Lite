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

class SequenceReverseComputeTester : public arena::TestCase {
 protected:
  std::string input_ = "X";
  std::string output_ = "Out";
  LoD lod_{{0, 2, 5, 10}};
  DDim dims_{{10, 4}};

 public:
  SequenceReverseComputeTester(const Place& place,
                               const std::string& alias,
                               LoD lod,
                               DDim dims)
      : TestCase(place, alias), lod_(lod), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindMutableTensor(input_);
    const auto* x_data = x->data<float>();
    (x->mutable_lod())->clear();
    for (size_t i = 0; i < lod_.size(); ++i) {
      (x->mutable_lod())->push_back(lod_[i]);
    }
    auto seq_offset = x->lod()[0];
    int width = x->numel() / dims_[0];
    auto* out = scope->NewTensor(output_);
    out->Resize(x->dims());
    (out->mutable_lod())->clear();
    for (size_t i = 0; i < lod_.size(); ++i) {
      (out->mutable_lod())->push_back(lod_[i]);
    }
    auto* out_data = out->mutable_data<float>();

    for (int i = 0; i < seq_offset.size() - 1; ++i) {
      auto start_pos = seq_offset[i];
      auto end_pos = seq_offset[i + 1];
      for (auto pos = start_pos; pos < end_pos; ++pos) {
        auto cur_pos = end_pos - pos - 1 + start_pos;
        std::memcpy(out_data + pos * width,
                    x_data + cur_pos * width,
                    width * sizeof(float));
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_reverse");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Y", {output_});
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (size_t i = 0; i < dims_.production(); ++i) {
      data[i] = i * 1.;
    }
    SetCommonTensor(input_, dims_, data.data(), lod_);
  }
};

void test_sequence_reverse(Place place) {
  DDim dims{{10, 4}};
  LoD lod{{0, 2, 3}, {0, 2, 5, 10}};
  std::unique_ptr<arena::TestCase> tester(
      new SequenceReverseComputeTester(place, "def", lod, dims));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

TEST(SequenceReverse, prec) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  test_sequence_reverse(place);
#endif
}

}  // namespace lite
}  // namespace paddle
