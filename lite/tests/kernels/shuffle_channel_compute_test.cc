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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <typename T>
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
    auto* out_data = out->template mutable_data<T>();

    auto* x = scope->FindTensor(input_);
    const auto* in_data = x->template data<T>();

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
          const T* p_i = in_data + (i * group_size + j) * spatial_size;
          T* p_o = out_data + (j * group_num + i) * spatial_size;
          memcpy(p_o, p_i, spatial_size * sizeof(T));
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
    std::vector<T> din(dims_.production());
    fill_data_rand<T>(din.data(), -1.0, 1.0, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

template <typename T>
void test_shuffle_channel(Place place, float abs_error = 2e-5) {
  for (int group : {2, 4, 8}) {
    std::unique_ptr<arena::TestCase> tester(
        new ShuffleChannelComputeTester<T>(place, "def", group));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(ShuffleChannel, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  test_shuffle_channel<float>(place, abs_error);
}

#ifdef ENABLE_ARM_FP16
TEST(ShuffleChannelFP16, precision) {
  Place place;
  place = Place(TARGET(kARM), PRECISION(kFP16));
  test_shuffle_channel<lite_api::float16_t>(place, 1e-4);
}
#endif

}  // namespace lite
}  // namespace paddle
