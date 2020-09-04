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

class FillConstantBatchSizeLikeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "input";
  std::string out_ = "out";
  DDim in_dims_{};
  LoD in_lod_{};
  std::vector<int> shape_{};
  float value_{0.f};
  int input_dim_idx_{0};
  int output_dim_idx_{0};
  int dtype_{static_cast<int>(VarDescAPI::VarDataType::FP32)};

 public:
  FillConstantBatchSizeLikeComputeTester(
      const Place& place,
      const std::string& alias,
      DDim in_dims,
      LoD in_lod,
      std::vector<int> shape,
      float value = 0.f,
      int input_dim_idx = 0,
      int output_dim_idx = 0,
      int dtype = static_cast<int>(VarDescAPI::VarDataType::FP32))
      : TestCase(place, alias),
        in_dims_(in_dims),
        in_lod_(in_lod),
        shape_(shape),
        value_(value),
        input_dim_idx_(input_dim_idx),
        output_dim_idx_(output_dim_idx),
        dtype_(dtype) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* input = scope->FindTensor(input_);
    std::vector<int64_t> out_shape{shape_.begin(), shape_.end()};
    if (input_dim_idx_ == 0 && !input->lod().empty()) {
      out_shape[output_dim_idx_] = input->lod().back().size() - 1;
    } else {
      out_shape[output_dim_idx_] = input->dims()[input_dim_idx_];
    }
    out->Resize(out_shape);

    auto* output_data = out->mutable_data<float>();
    for (int i = 0; i < out->numel(); i++) {
      output_data[i] = value_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fill_constant_batch_size_like");
    op_desc->SetInput("Input", {input_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("shape", shape_);
    op_desc->SetAttr("value", value_);
    op_desc->SetAttr("input_dim_idx", input_dim_idx_);
    op_desc->SetAttr("output_dim_idx", output_dim_idx_);
    op_desc->SetAttr("dtype", dtype_);
  }

  void PrepareData() override {
    std::vector<float> din(in_dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, in_dims_.production());
    SetCommonTensor(input_, in_dims_, din.data(), in_lod_);
  }
};

void TestFillConstantBatchSizeLike(Place place, float abs_error) {
  for (auto input_dim_idx : {0, 1, 2}) {
    for (auto output_dim_idx : {0, 1, 2}) {
      std::unique_ptr<arena::TestCase> tester(
          new FillConstantBatchSizeLikeComputeTester(place,
                                                     "def",
                                                     DDim{{5, 4, 3}},
                                                     {},
                                                     {2, 3, 4},
                                                     0.f,
                                                     input_dim_idx,
                                                     output_dim_idx));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestFillConstantBatchSizeLikeLod(Place place, float abs_error) {
  for (auto lod : std::vector<LoD>{{{0, 1, 4, 5}}, {{0, 2, 4}, {0, 1, 4, 5}}}) {
    std::unique_ptr<arena::TestCase> tester(
        new FillConstantBatchSizeLikeComputeTester(
            place, "def", DDim{{5, 4, 3}}, lod, {2, 3, 4}, 0.f));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

void TestFillConstantBatchSizeLikeValue(Place place, float abs_error) {
  std::vector<float> values{-1., 3.5};
  for (auto value : values) {
    std::unique_ptr<arena::TestCase> tester(
        new FillConstantBatchSizeLikeComputeTester(
            place, "def", DDim{{5, 4, 3}}, {}, {2, 3}, value));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(fill_constant_batch_size_like, precision) {
  LOG(INFO) << "test fill_constant_batch_size_like op";
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestFillConstantBatchSizeLike(place, abs_error);
  TestFillConstantBatchSizeLikeLod(place, abs_error);
  TestFillConstantBatchSizeLikeValue(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
