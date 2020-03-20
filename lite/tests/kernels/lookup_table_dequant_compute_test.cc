// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

void dequant(const unsigned char* in,
             float* out,
             float min,
             float max,
             int emb_size,
             int pow_2_bits) {
  float scale = (max - min) / pow_2_bits;
  for (int i = 0; i < emb_size; ++i) {
    float x = scale * static_cast<int>(in[i]) + min;
    out[i] = x;
  }
}

class LookupTableDequantComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "lookup_table_dequant";
  std::string ids_ = "ids";
  std::string w_ = "w";
  std::string out_ = "out";
  DDim ids_dims_{{2, 1}};
  DDim w_dims_{{8, 4}};
  int64_t padding_idx_ = -1;

 public:
  LookupTableDequantComputeTest(const Place& place,
                                const std::string& alias,
                                const DDim& ids_dims,
                                const DDim& w_dims,
                                int64_t padding_idx)
      : TestCase(place, alias),
        ids_dims_(ids_dims),
        w_dims_(w_dims),
        padding_idx_(padding_idx) {}

  void RunBaseline(Scope* scope) override {
    auto ids = scope->FindTensor(ids_);
    auto w = scope->FindTensor(w_);
    auto ids_dims = ids->dims();
    auto w_dims = w->dims();

    auto out = scope->NewTensor(out_);
    CHECK(out);

    int ids_rank = ids_dims.size();
    CHECK_EQ(ids_dims[ids_rank - 1], 1);
    CHECK_EQ(w_dims.size(), 2);

    std::vector<int64_t> out_dims;
    for (int i = 0; i < ids_rank - 1; ++i) {
      out_dims.push_back(ids_dims[i]);
    }
    out_dims.push_back((w_dims[1] - 2) * 4);
    out->Resize(out_dims);
    out->set_lod(ids->lod());

    auto ids_data = ids->data<int64_t>();
    auto ids_size = ids_dims.production();
    auto w_data = w->data<float>();
    auto w_rows = w_dims[0];
    auto quant_number = w_dims[1];
    auto w_cols = (quant_number - 2) * 4;
    auto out_data = out->mutable_data<float>();
    int pow_2_bits = static_cast<int>(pow(2, 8));

    for (int64_t i = 0; i < ids_size; i++) {
      auto id = ids_data[i];
      if (padding_idx_ != -1 && id == padding_idx_) {
        memset(out_data + i * w_cols, 0, w_cols * sizeof(float));
      } else {
        CHECK_LT(id, w_rows) << "lookup_table ids[i] expected < " << w_rows
                             << " but got " << id;
        CHECK_GE(id, 0) << "lookup_table ids[i] expected >= 0 but got " << id;
        float min = *(w_data + ids_data[i] * quant_number);
        float max = *(w_data + ids_data[i] * quant_number + 1);
        int offset = ids_data[i] * quant_number + 2;
        const unsigned char* tensor_buf =
            reinterpret_cast<const unsigned char*>(w_data + offset);
        dequant(
            tensor_buf, out_data + i * w_cols, min, max, w_cols, pow_2_bits);
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("Ids", {ids_});
    op_desc->SetInput("W", {w_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr<int64_t>("padding_idx", padding_idx_);
  }

  void PrepareData() override {
    std::vector<int64_t> ids(ids_dims_.production());
    fill_data_rand<int64_t>(
        ids.data(), 0, w_dims_[0] - 1, ids_dims_.production());

    std::vector<float> w(w_dims_.production());
    fill_data_rand(w.data(), -1.f, 1.f, w_dims_.production());

    SetCommonTensor(ids_, ids_dims_, ids.data());
    SetCommonTensor(w_, w_dims_, w.data());
  }
};

TEST(LookupTableDequant, precision) {
#ifdef LITE_WITH_ARM
  float abs_error = 2e-5;
  Place place = TARGET(kARM);
  for (auto ids_dims :
       std::vector<std::vector<int64_t>>{{5, 2, 3, 1}, {2, 3, 1}, {3, 1}}) {
    for (auto w_dims :
         std::vector<std::vector<int64_t>>{{4, 3}, {6, 8}, {12, 15}}) {
      for (auto padding_idx : std::vector<int64_t>{-1}) {
        std::unique_ptr<arena::TestCase> tester(
            new LookupTableDequantComputeTest(
                place, "def", DDim(ids_dims), DDim(w_dims), padding_idx));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
