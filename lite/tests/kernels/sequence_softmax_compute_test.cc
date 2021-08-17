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

class SequenceSoftmaxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  DDim dims_{{7, 1}};
  LoD lod_{{0, 7}};
  int seq_num_ = 1;

 public:
  SequenceSoftmaxComputeTester(const Place& place,
                               const std::string& alias,
                               LoD lod)
      : TestCase(place, alias), lod_(lod) {
    DDim dims{{int64_t(lod[0].back()), int64_t(1)}};
    dims_ = dims;
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindMutableTensor(input_);
    const auto* x_data = x->data<float>();
    (x->mutable_lod())->clear();
    (x->mutable_lod())->push_back(lod_[0]);
    auto seq_offset = x->lod()[0];
    int in_h = dims_[0];
    int in_w = x->numel() / in_h;
    CHECK_EQ(in_w, 1) << "input dims is not valid";
    int seq_num = seq_offset.size() - 1;
    for (int i = 0; i < seq_num; i++) {
      float seq_max = x_data[seq_offset[i]];
      float exp_sum = 0.f;
      for (size_t j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
        seq_max = std::max(seq_max, x_data[j]);
      }
      for (size_t j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
        exp_sum += expf(x_data[j] - seq_max);
      }
      for (size_t j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
        out_data[j] = expf(x_data[j] - seq_max) / exp_sum;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_softmax");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = 0.1;
    }

    SetCommonTensor(input_, dims_, data.data(), lod_);
  }
};

void generate_lod(int seq_num,
                  int max_len,
                  std::vector<uint64_t>& seq_offset) {  // NOLINT
  seq_offset.clear();
  int sum = 0;
  seq_offset.push_back(sum);
  for (int i = 0; i < seq_num; i++) {
    sum += std::rand() % max_len + 1;
    seq_offset.push_back(uint64_t(sum));
  }
}

void test_sequence_softmax(Place place) {
  int max_len = 10;
  for (int seq_num : {1, 3, 5}) {
    std::vector<std::vector<uint64_t>> lod;
    lod.resize(1);
    generate_lod(seq_num, max_len, lod[0]);
    std::unique_ptr<arena::TestCase> tester(
        new SequenceSoftmaxComputeTester(place, "def", lod));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

TEST(SequenceSoftmax, precision) {
  Place place;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  test_sequence_softmax(place);
}

}  // namespace lite
}  // namespace paddle
