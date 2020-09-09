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

class GatherComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "gather";
  std::string x_ = "x";
  std::string index_ = "index";
  std::string out_ = "out";
  DDim x_dims_{{5, 4, 2, 3}};
  DDim index_dims_{{2, 1}};

 public:
  GatherComputeTest(const Place& place,
                    const std::string& alias,
                    const DDim& x_dims,
                    const DDim& index_dims)
      : TestCase(place, alias), x_dims_(x_dims), index_dims_(index_dims) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto index = scope->FindTensor(index_);
    auto x_dims = x->dims();
    auto index_dims = index->dims();
    CHECK(index_dims.size() == 1 ||
          (index_dims.size() == 2 && index_dims[1] == 1));

    auto out = scope->NewTensor(out_);
    CHECK(out);
    int batch_size = index_dims[0];
    DDim out_dims = x_dims;
    out_dims[0] = batch_size;
    out->Resize(out_dims);

    auto x_data = x->data<float>();
    auto index_data = index->data<int>();
    auto out_data = out->mutable_data<float>();

    auto slice_num = x_dims[0];
    auto slice_size = x_dims.Slice(1, x_dims.size()).production();
    for (int i = 0; i < batch_size; i++) {
      auto index = index_data[i];
      CHECK_LT(index, slice_num) << "gather index[i] expected < " << slice_num
                                 << " but got " << index;
      CHECK_GE(index, 0) << "gather ids[i] expected >= 0 but got " << index;
      memcpy(out_data + i * slice_size,
             x_data + index * slice_size,
             slice_size * sizeof(float));
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Index", {index_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());

    std::vector<int32_t> index(index_dims_.production());
    fill_data_rand<int32_t>(
        index.data(), 0, x_dims_[0] - 1, index_dims_.production());

    SetCommonTensor(x_, x_dims_, x.data());
    SetCommonTensor(index_, index_dims_, index.data());
  }
};

TEST(Gather, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#else
  return;
#endif

  for (auto x_dims :
       std::vector<std::vector<int64_t>>{{5, 2, 3, 4}, {8, 3, 5}, {12, 3}}) {
    for (auto index_dims : std::vector<std::vector<int64_t>>{{3}, {7}, {10}}) {
      std::unique_ptr<arena::TestCase> tester(
          new GatherComputeTest(place, "def", DDim(x_dims), DDim(index_dims)));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

}  // namespace lite
}  // namespace paddle
