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
#include <cmath>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class DropoutComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string type_ = "dropout";
  std::string x_ = "x";
  std::string out_ = "out";
  std::string mask_ = "mask";
  DDim dims_{{1}};
  float dropout_prob_ = 0.5;
  bool fix_seed_ = true;
  int seed_ = 1;
  std::string dropout_implementation_ = "downgrade_in_infer";
  int is_test_ = 1;

 public:
  DropoutComputeTester(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       float dropout_prob,
                       std::string dropout_implementation)
      : TestCase(place, alias),
        dims_(dims),
        dropout_prob_(dropout_prob),
        dropout_implementation_(dropout_implementation) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(dims_);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->data<float>();

    if (dropout_implementation_ == "downgrade_in_infer") {
      float rate = 1 - dropout_prob_;
      for (int64_t i = 0; i < dims_.production(); i++) {
        output_data[i] = x_data[i] * rate;
      }
    } else if (dropout_implementation_ == "upscale_in_train") {
      memcpy(output_data, x_data, sizeof(float) * dims_.production());
    } else {
      LOG(FATAL) << "unsupported dropout_implementation: "
                 << dropout_implementation_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    if (!is_test_) {
      op_desc->SetOutput("Mask", {mask_});
    }
    op_desc->SetAttr("dropout_prob", dropout_prob_);
    op_desc->SetAttr("fix_seed", fix_seed_);
    op_desc->SetAttr("seed", seed_);
    op_desc->SetAttr("dropout_implementation", dropout_implementation_);
    op_desc->SetAttr("is_test", is_test_);
  }

  void PrepareData() override {
    std::vector<float> x(dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(x_, dims_, x.data());
  }
};

TEST(Dropout, precision) {
  LOG(INFO) << "test dropout op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {3}, {3, 4}, {3, 4, 5}, {1, 2, 3, 4}, {2, 3, 4, 5}}) {
    for (auto dropout_prob : {0., 0.5, 1.}) {
      for (auto dropout_implementation :
           {"downgrade_in_infer", "upscale_in_train"}) {
#ifdef LITE_WITH_NPU
        if (dims.size() < 2) continue;
#endif
        std::unique_ptr<arena::TestCase> tester(new DropoutComputeTester(
            place, "def", DDim(dims), dropout_prob, dropout_implementation));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision({"mask"});
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
