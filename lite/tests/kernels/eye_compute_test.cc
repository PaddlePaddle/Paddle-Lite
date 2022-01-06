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

class EyeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string out_ = "out";

  int64_t num_rows_{0};
  int64_t num_columns_{0};
  int dtype_{static_cast<int>(VarDescAPI::VarDataType::FP32)};

 public:
  EyeComputeTester(const Place& place,
                   const std::string& alias,
                   int64_t num_rows,
                   int64_t num_columns,
                   int dtype)
      : TestCase(place, alias),
        num_rows_(num_rows),
        num_columns_(num_columns),
        dtype_(dtype) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    std::vector<int64_t> out_shape{num_rows_, num_columns_};
    out->Resize(out_shape);

    switch (dtype_) {
      case static_cast<int>(VarDescAPI::VarDataType::BOOL): {
        MakeIdentityMatrix<bool>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::INT16): {
        MakeIdentityMatrix<int16_t>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::INT32): {
        MakeIdentityMatrix<int32_t>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::INT64): {
        MakeIdentityMatrix<int64_t>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::FP32): {
        MakeIdentityMatrix<float>(scope);
        break;
      }
      default: {
        LOG(FATAL) << "Attribute dtype in fill_constant op test"
                      "must be 0[bool] or 1[int16] or 2[int32] or "
                      "3[int64] or 5[fp32] for baseline: "
                   << dtype_;
        break;
      }
    }
  }

  template <typename T>
  void MakeIdentityMatrix(Scope* scope) {
    auto* out = scope->FindMutableTensor(out_);
    auto* output_data = out->mutable_data<T>();
    T value = static_cast<T>(0);
    for (int i = 0; i < out->numel(); i++) {
      output_data[i] = value;
    }
    for (int i = 0; i < num_rows_; i++) {
      for (int j = 0; j < num_columns_; j++) {
        if (i == j) output_data[i * num_columns_ + j] = static_cast<T>(1);
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("eye");
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("num_rows", num_rows_);
    op_desc->SetAttr("num_columns", num_columns_);
    op_desc->SetAttr("dtype", dtype_);
  }

  void PrepareData() override {}
};

void TestEyeShape(Place place, float abs_error) {
  std::vector<std::vector<int64_t>> out_shapes{{2, 2}, {4, 4}, {3, 4}, {4, 3}};
  for (auto out_shape : out_shapes) {
    std::unique_ptr<arena::TestCase> tester(
        new EyeComputeTester(place,
                             "def",
                             out_shape[0],
                             out_shape[1],
                             static_cast<int>(VarDescAPI::VarDataType::FP32)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

void TestEyeDtype(Place place, float abs_error) {
  std::vector<int> dtypes{0, 1, 2, 3, 5};
  for (auto dtype : dtypes) {
    std::unique_ptr<arena::TestCase> tester(
        new EyeComputeTester(place, "def", 3, 3, dtype));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(fill_constant, precision) {
  Place place;
  float abs_error = 1e-5;
  place = TARGET(kHost);

  TestEyeShape(place, abs_error);
  TestEyeDtype(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
