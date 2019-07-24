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
#include "lite/api/paddle_use_kernels.h"  // 需要修改这个文件加入我们新写的 kernels
#include "lite/api/paddle_use_ops.h"  // 需要修改这个文件加入我们新写的 OP
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

class NegativeComputeTester : public arena::TestCase {
 protected:  // 定义变量
  // common attributes for this op.
  //  std::string input_ = "x";
  //  std::string output_ = "out";
  std::string input_ = "X";
  std::string output_ = "Out";
  DDim dims_{{100, 20}};
  /*   float scale_ = 0.;
    float bias_ = 0.;
    DDim dims_{{100, 20}};
    bool bias_after_scale_;*/

 public:
  NegativeComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}  // modified

  void RunBaseline(Scope* scope) override {
    LOG(INFO) << "1";
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    LOG(INFO) << "2";
    auto* out_data = out->mutable_data<float>();
    LOG(INFO) << "3";
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    LOG(INFO) << "4";
    for (int i = 0; i < dims_.production(); i++) {
      out_data[i] = -x_data[i];
      LOG(INFO) << "i" << i;
    }
    LOG(INFO) << "5";
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("negative");  // 我们的op的名字
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {  // 给X赋初始值
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

TEST(Negative, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif
  std::unique_ptr<arena::TestCase> tester(
      new NegativeComputeTester(place, "def"));  // modified
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

}  // namespace lite
}  // namespace paddle
