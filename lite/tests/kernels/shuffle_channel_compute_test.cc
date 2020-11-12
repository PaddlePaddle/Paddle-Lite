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

class ShuffleChannelComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  int group_ = 4;
  DDim dims_{{10, 16, 4, 4}};

 public:
  ShuffleChannelComputeTester(const Place& place,
                              const std::string& alias,
                              int group)
      : TestCase(place, alias), group_(group) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* in_data = x->data<float>();

    int num = dims_[0];
    int channel = dims_[1];
    int height = dims_[2];
    int width = dims_[3];
    int feather_size = channel * height * width;
    int spatial_size = height * width;
    int group_num = group_;
    int group_size = channel / group_;
    for (int n = 0; n < num; n++) {
      for (int i = 0; i < group_num; ++i) {
        for (int j = 0; j < group_size; ++j) {
          const float* p_i = in_data + (i * group_size + j) * spatial_size;
          float* p_o = out_data + (j * group_num + i) * spatial_size;
          memcpy(p_o, p_i, spatial_size * sizeof(float));
        }
      }
      in_data += feather_size;
      out_data += feather_size;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("shuffle_channel");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("group", group_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

void test_shuffle_channel(Place place, float abs_error = 2e-5) {
  for (int group : {2, 4, 8}) {
    std::unique_ptr<arena::TestCase> tester(
        new ShuffleChannelComputeTester(place, "def", group));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(ShuffleChannel, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#else
  return;
#endif

  test_shuffle_channel(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
