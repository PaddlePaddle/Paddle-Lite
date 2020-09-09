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

namespace paddle {
namespace lite {

class BoxClipComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "input";
  std::string im_info_ = "im_info";
  std::string output_ = "output";
  DDim input_dims_{};
  LoD input_lod_{};
  DDim im_info_dim_{};

 public:
  BoxClipComputeTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {
    input_dims_.ConstructFrom(std::vector<int64_t>({4, 3, 4}));
    std::vector<uint64_t> lod0 = {0, 1, 4};
    input_lod_.push_back(lod0);
    im_info_dim_.ConstructFrom(std::vector<int64_t>({2, 3}));
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(input_dims_);
    auto* out_lod = out->mutable_lod();
    *out_lod = input_lod_;
    auto* out_data = out->mutable_data<float>();

    auto* input = scope->FindTensor(input_);
    const auto* input_data = input->data<float>();
    for (int i = 0; i < 12; i++) {
      out_data[i] = std::max(std::min(input_data[i], 9.f), 0.f);
    }
    for (int i = 12; i < 48; i++) {
      out_data[i] = std::max(std::min(input_data[i], 14.f), 0.f);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("box_clip");
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("ImInfo", {im_info_});
    op_desc->SetOutput("Output", {output_});
  }

  void PrepareData() override {
    std::vector<float> input_data(input_dims_.production());
    for (int i = 0; i < input_dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      input_data[i] = sign * static_cast<float>((i * 7) % 20);
    }
    SetCommonTensor(input_, input_dims_, input_data.data(), input_lod_);

    std::vector<float> im_info_data{10, 10, 1, 15, 15, 1};
    SetCommonTensor(im_info_, im_info_dim_, im_info_data.data());
  }
};

TEST(Boxclip, precision) {
  LOG(INFO) << "test box_clip op";
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  std::unique_ptr<arena::TestCase> tester(
      new BoxClipComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

}  // namespace lite
}  // namespace paddle
