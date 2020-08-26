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

template <typename T>
class LookupTableComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "lookup_table";
  std::string ids_ = "ids";
  std::string w_ = "w";
  std::string out_ = "out";
  DDim ids_dims_{{2, 1}};
  DDim w_dims_{{8, 4}};
  int64_t padding_idx_ = -1;

 public:
  LookupTableComputeTest(const Place& place,
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
    out_dims.push_back(w_dims[1]);
    out->Resize(out_dims);
    out->set_lod(ids->lod());

    auto ids_data = ids->template data<T>();
    auto ids_size = ids_dims.production();
    auto w_data = w->template data<float>();
    auto w_rows = w_dims[0];
    auto w_cols = w_dims[1];
    auto out_data = out->template mutable_data<float>();

    for (int64_t i = 0; i < ids_size; i++) {
      auto id = ids_data[i];
      if (padding_idx_ != -1 && id == padding_idx_) {
        memset(out_data + i * w_cols, 0, w_cols * sizeof(float));
      } else {
        CHECK_LT(id, w_rows) << "lookup_table ids[i] expected < " << w_rows
                             << " but got " << id;
        CHECK_GE(id, 0) << "lookup_table ids[i] expected >= 0 but got " << id;
        memcpy(out_data + i * w_cols,
               w_data + id * w_cols,
               w_cols * sizeof(float));
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
    std::vector<T> ids(ids_dims_.production());
    fill_data_rand<T>(ids.data(), 0, w_dims_[0] - 1, ids_dims_.production());

    std::vector<float> w(w_dims_.production());
    fill_data_rand(w.data(), -1.f, 1.f, w_dims_.production());

    SetCommonTensor(ids_, ids_dims_, ids.data());
    SetCommonTensor(w_, w_dims_, w.data());
  }
};

TEST(LookupTable, precision) {
  LOG(INFO) << "test lookup_table op";
  float abs_error = 1e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

#if defined(LITE_WITH_NPU)
  using ID_T = int;
#else
  using ID_T = int64_t;
#endif

  for (auto ids_dims :
       std::vector<std::vector<int64_t>>{{5, 2, 3, 1}, {2, 3, 1}, {3, 1}}) {
    for (auto w_dims :
         std::vector<std::vector<int64_t>>{{4, 2}, {6, 8}, {12, 15}}) {
#if (defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)) || \
    defined(LITE_WITH_NPU)
      for (auto padding_idx :
           std::vector<int64_t>{-1}) {  // Only -1 is supported by XPU or NPU
#else
      for (auto padding_idx : std::vector<int64_t>{-1, 0, w_dims[0] - 1}) {
#endif
        std::unique_ptr<arena::TestCase> tester(
            new LookupTableComputeTest<ID_T>(
                place, "def", DDim(ids_dims), DDim(w_dims), padding_idx));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
