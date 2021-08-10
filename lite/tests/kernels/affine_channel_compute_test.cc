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

class AffineChannelComputeTester : public arena::TestCase {
 protected:
  std::string input_ = "x";
  std::string scale_ = "scale";
  std::string bias_ = "bias";
  std::string output_ = "out";
  std::string data_layout_ = "";
  DDim x_dims_{{2, 5, 20, 30}};

 public:
  AffineChannelComputeTester(const Place& place,
                             const std::string& alias,
                             int n,
                             int c,
                             int h,
                             int w,
                             std::string data_layout)
      : TestCase(place, alias) {
    data_layout_ = data_layout;
    CHECK(data_layout_ == "NCHW" || data_layout == "NHWC");
    if (data_layout_ == "NCHW") {
      x_dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
    } else if (data_layout_ == "NHWC") {
      x_dims_ = DDim(std::vector<int64_t>({n, h, w, c}));
    }
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(x_dims_);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    auto* scale = scope->FindTensor(scale_);
    const auto* scale_data = scale->data<float>();
    auto* bias = scope->FindTensor(bias_);
    const auto* bias_data = bias->data<float>();

    int num = x_dims_[0];

    if (data_layout_ == "NCHW") {
      int channel = x_dims_[1];
      int size = x_dims_[2] * x_dims_[3];
      int in_channel = channel * size;
      for (int n = 0; n < num; n++) {
        auto x_data_n = x_data + n * in_channel;
        auto output_data_n = output_data + n * in_channel;
        for (int c = 0; c < channel; c++) {
          auto x_data_c = x_data_n + c * size;
          auto output_data_c = output_data_n + c * size;
          for (int k = 0; k < size; k++) {
            output_data_c[k] = scale_data[c] * x_data_c[k] + bias_data[c];
          }
        }
      }
    } else if (data_layout_ == "NHWC") {
      int channel = x_dims_[3];
      int height = x_dims_[1];
      int width = x_dims_[2];
      int hwc = height * width * channel;
      int wc = width * channel;
      for (int n = 0; n < num; n++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            auto x_ptr = x_data + n * hwc + h * wc + w * channel;
            auto output_ptr = output_data + n * hwc + h * wc + w * channel;
            for (int c = 0; c < channel; c++) {
              output_ptr[c] = x_ptr[c] * scale_data[c] + bias_data[c];
            }
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("affine_channel");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Scale", {scale_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetAttr("data_layout", data_layout_);
    op_desc->SetOutput("Out", {output_});
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(input_, x_dims_, x_data.data());

    int c = data_layout_ == "NCHW" ? x_dims_[1] : x_dims_[3];
    DDim scale_dims(std::vector<int64_t>({c}));
    std::vector<float> scale_data(scale_dims.production());
    for (int i = 0; i < scale_dims.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      scale_data[i] = sign * static_cast<float>(i % 128) * 0.005f + 0.001;
    }
    SetCommonTensor(scale_, scale_dims, scale_data.data());

    DDim bias_dims(std::vector<int64_t>({c}));
    std::vector<float> bias_data(bias_dims.production());
    for (int i = 0; i < bias_dims.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      bias_data[i] = sign * static_cast<float>(i % 128) * 0.005f + 0.001;
    }
    SetCommonTensor(bias_, bias_dims, bias_data.data());
  }
};

TEST(AffineChannel, precision) {
  LOG(INFO) << "test affine_channel op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (int n : {1, 5}) {
    for (int c : {2, 5}) {
      for (int h : {3, 10}) {
        for (int w : {3, 10}) {
          for (std::string data_layout : {"NCHW", "NHWC"}) {
            std::unique_ptr<arena::TestCase> tester(
                new AffineChannelComputeTester(
                    place, "def", n, c, h, w, data_layout));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
