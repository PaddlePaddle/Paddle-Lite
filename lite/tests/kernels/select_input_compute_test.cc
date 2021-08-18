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

DDim infer_shape(const std::vector<const Tensor*>& inputs, int mask) {
  DDim out_dims;
  out_dims = inputs[mask]->dims();

  return out_dims;
}

class SelectInputComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::vector<std::string> x_vct_{};
  std::string out_ = "out";
  std::string mask_ = "mask";

  int x_num_ = 3;
  DDim x_dims_{{2, 3, 4, 5}};
  DDim mask_dim{{1}};

 public:
  SelectInputComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    std::vector<const Tensor*> x_vct;
    for (std::string& name : x_vct_) {
      x_vct.push_back(scope->FindTensor(name));
    }
    const Tensor* Mask = scope->FindTensor(mask_);
    auto* out = scope->NewTensor(out_);
    int Mask_ = Mask->data<int>()[0];
    DDim output_dims = infer_shape(x_vct, Mask_);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();
    auto* in_data = x_vct[Mask_]->data<float>();
    for (int i = 0; i < x_vct[Mask_]->data_size(); i++) {
      output_data[i] = in_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("select_input");
    op_desc->SetInput("X", x_vct_);
    op_desc->SetInput("Mask", {mask_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    for (int n = 0; n < x_num_; n++) {
      std::vector<float> x_data(x_dims_.production());
      for (int i = 0; i < x_dims_.production(); i++) {
        x_data[i] = static_cast<float>(i + n);
      }
      const std::string x_name = "x_tensor_" + paddle::lite::to_string(n);
      x_vct_.push_back(x_name);
      SetCommonTensor(x_name, x_dims_, x_data.data());
    }
    std::string mask_name = "mask";
    std::vector<int> mask_data = {0};
    SetCommonTensor(mask_name, mask_dim, mask_data.data());
  }
};

TEST(SelectInput, precision) {
  LOG(INFO) << "test SelectInput op, kHost";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif
  std::unique_ptr<arena::TestCase> tester(
      new SelectInputComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

}  // namespace lite
}  // namespace paddle
