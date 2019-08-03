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

class ShuffleChannelComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  int group_ = 1;
  DDim dims_{{1, 2}};

 public:
  ShuffleChannelComputeTester(const Place& place,
                              const std::string& alias,
                              int group)
      : TestCase(place, alias), group_(group) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* outputs = out->mutable_data<float>();
    auto* x = scope->FindTensor(input_);
    const auto* inputs = x->data<float>();
    DDim x_dims = x->dims();
    int num = x_dims.production();
    int channel = x->dims()[1];
    int height = x->dims()[2];
    int width = x->dims()[3];
    int fea_size = channel * height * width;
    int spatial_size = height * width;
    int group_row = group_;
    int group_col = channel / group_;
    for (int k = 0; k < num; ++k) {
      inputs += k * fea_size;
      outputs += k * fea_size;
      for (int i = 0; i < group_row; ++i) {
        for (int j = 0; j < group_col; ++j) {
          const float* p_i = inputs + (i * group_col + j) * spatial_size;
          float* p_o = outputs + (j * group_row + i) * spatial_size;
          memcpy(p_o, p_i, spatial_size * sizeof(float));
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("shuffle_channel");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("group", group_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

void test_shuffle_channel(Place place) {
  for (int group : {1, 2, 3}) {
    std::unique_ptr<arena::TestCase> tester(
        new ShuffleChannelComputeTester(place, "def", group));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

TEST(ShuffleChannel, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_shuffle_channel(place);
#endif
}

}  // namespace lite
}  // namespace paddle
