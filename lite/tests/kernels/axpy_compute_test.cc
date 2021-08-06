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

class AxpyComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::string scale_ = "scale";
  std::string bias_ = "bias";
  DDim x_dims_{{2, 5, 20, 30}};

 public:
  AxpyComputeTester(
      const Place& place, const std::string& alias, int n, int c, int h, int w)
      : TestCase(place, alias) {
    x_dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    std::vector<int64_t> output_shape(
        {x_dims_[0], x_dims_[1], x_dims_[2], x_dims_[3]});
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    auto* scale = scope->FindTensor(scale_);
    const auto* scale_data = scale->data<float>();
    auto* bias = scope->FindTensor(bias_);
    const auto* bias_data = bias->data<float>();

    int num = x_dims_[0];
    int channel = x_dims_[1];
    int size = x_dims_[2] * x_dims_[3];
    int in_channel = channel * size;

    for (int i = 0; i < num; i++) {
      auto scale_data_i = scale_data + i * channel;
      auto x_data_i = x_data + i * in_channel;
      auto bias_data_i = bias_data + i * in_channel;
      auto output_data_i = output_data + i * in_channel;
      for (int j = 0; j < channel; j++) {
        auto scale_data_j = scale_data_i + j;
        auto x_data_j = x_data_i + j * size;
        auto bias_data_j = bias_data_i + j * size;
        auto output_data_j = output_data_i + j * size;
        for (int k = 0; k < size; k++) {
          output_data_j[k] = scale_data_j[0] * x_data_j[k] + bias_data_j[k];
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("axpy");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Scale", {scale_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(input_, x_dims_, x_data.data());

    int n = x_dims_[0];
    int c = x_dims_[1];
    int h = x_dims_[2];
    int w = x_dims_[3];
    DDim scale_dims(std::vector<int64_t>({n, c}));
    std::vector<float> scale_data(scale_dims.production());
    for (int i = 0; i < scale_dims.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      scale_data[i] = sign * static_cast<float>(i % 128) * 0.005f + 0.001;
    }
    SetCommonTensor(scale_, scale_dims, scale_data.data());

    DDim bias_dims(std::vector<int64_t>({n, c, h, w}));
    std::vector<float> bias_data(bias_dims.production());
    for (int i = 0; i < bias_dims.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      bias_data[i] = sign * static_cast<float>(i % 128) * 0.005f + 0.001;
    }
    SetCommonTensor(bias_, bias_dims, bias_data.data());
  }
};

TEST(Axpy, precision) {
  LOG(INFO) << "test axpy op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (int n : {1, 3}) {
    for (int c : {3, 6}) {
      for (int h : {9, 18}) {
        for (int w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(
              new AxpyComputeTester(place, "def", n, c, h, w));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
