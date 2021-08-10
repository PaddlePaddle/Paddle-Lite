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

class SequenceExpandComputeTester : public arena::TestCase {
 protected:
  const std::string input_x_ = "x";
  const std::string input_y_ = "y";
  const std::string output_ = "out";
  LoD lod_x_{{0, 2, 4}};
  LoD lod_y_{{0, 1, 4}};
  int ref_level_ = -1;
  DDim dims_{{4, 1}};

 public:
  SequenceExpandComputeTester(const Place& place,
                              const std::string& alias,
                              LoD lod_x,
                              LoD lod_y,
                              int ref_level,
                              DDim dims)
      : TestCase(place, alias),
        lod_x_(lod_x),
        lod_y_(lod_y),
        ref_level_(ref_level),
        dims_(dims) {}

  void RunBaseline(Scope* scope) {
    auto* out = scope->NewTensor(output_);

    auto* x = scope->FindMutableTensor(input_x_);
    const auto* x_data = x->data<float>();
    (x->mutable_lod())->clear();
    (x->mutable_lod())->push_back(lod_x_[0]);
    auto width = x->numel() / dims_[0];
    auto lod_x = x->lod();

    auto* y = scope->FindMutableTensor(input_y_);
    (y->mutable_lod())->clear();
    for (int i = 0; i < lod_y_.size(); i++) {
      (y->mutable_lod())->push_back(lod_y_[i]);
    }
    if (ref_level_ == -1) {
      ref_level_ = lod_y_.size() - 1;
    }
    auto lod_y = y->lod()[ref_level_];

    DDim out_dims(dims_);
    int64_t out_first_dim = 0;
    if (lod_x.size() > 0) {
      if (lod_y.size() <= 1) {
        out_first_dim = dims_[0];
      } else {
        for (int i = 1; i < lod_y.size(); ++i) {
          int64_t x_seq_len = 1;
          if (lod_x.size() == 1) {
            x_seq_len = lod_x[0][i] - lod_x[0][i - 1];
          }
          out_first_dim += (lod_y[i] - lod_y[i - 1]) * x_seq_len;
        }
        out_dims[0] = out_first_dim;
      }
    } else {
      out_dims[0] = -1;
    }
    out->Resize(out_dims);

    auto* out_data = out->mutable_data<float>();
    if (lod_x.size() == 0) {
      for (int i = 0; i < lod_y.size() - 1; i++) {
        for (int j = lod_y[i]; j < lod_y[i + 1]; j++) {
          memcpy(
              out_data + j * width, x_data + i * width, sizeof(float) * width);
        }
      }
      (out->mutable_lod())->push_back(lod_y);
    } else {
      std::vector<uint64_t> output_lod;
      output_lod.push_back(0);
      uint64_t offset = 0;
      uint64_t out_offset = 0;
      for (int i = 0; i < lod_y.size() - 1; i++) {
        auto x_seq_len = lod_x[0][i + 1] - lod_x[0][i];
        auto repeat = lod_y[i + 1] - lod_y[i];
        for (int j = 0; j < repeat; j++) {
          for (int k = 0; k < x_seq_len; k++) {
            memcpy(out_data + (offset + j * x_seq_len + k) * width,
                   x_data + (lod_x[0][i] + k) * width,
                   width * sizeof(float));
          }
          out_offset += x_seq_len;
          output_lod.push_back(out_offset);
        }
        offset += repeat * x_seq_len;
      }
      (out->mutable_lod())->push_back(output_lod);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_expand");
    op_desc->SetInput("X", {input_x_});
    op_desc->SetInput("Y", {input_y_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("ref_level", ref_level_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }
    SetCommonTensor(input_x_, dims_, data.data(), lod_x_);
    SetCommonTensor(input_y_, dims_, data.data(), lod_y_);
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

void test_sequence_expand(Place place) {
  int max_len = 2;
  for (int ref_level : {-1, 0}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (int seq_num : {1, 3, 5}) {
            std::vector<std::vector<uint64_t>> lod_x;
            std::vector<std::vector<uint64_t>> lod_y;
            lod_x.resize(1);
            lod_y.resize(1);
            generate_lod(seq_num, max_len, lod_x[0]);
            generate_lod(seq_num, max_len, lod_y[0]);
            int n = int64_t(lod_x[0].back());
            auto dims_x = DDim(std::vector<int64_t>({n, c, h, w}));
            std::unique_ptr<arena::TestCase> tester(
                new SequenceExpandComputeTester(
                    place, "def", lod_x, lod_y, ref_level, dims_x));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(SequenceExpand, precision) {
  Place place;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  test_sequence_expand(place);
}

}  // namespace lite
}  // namespace paddle
