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

class AnchorGeneratorComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_str_ = "Input";
  std::string anchors_str_ = "Anchors";
  std::string variances_str_ = "Variances";
  DDim input_dims_;
  std::vector<float> anchor_sizes_;
  std::vector<float> aspect_ratios_;
  std::vector<float> stride_;
  std::vector<float> variances_;
  float offset_;

 public:
  AnchorGeneratorComputeTester(const Place& place,
                               const std::string& alias,
                               int n,
                               int c,
                               int h,
                               int w,
                               std::vector<float> anchor_sizes,
                               std::vector<float> aspect_ratios,
                               std::vector<float> stride,
                               std::vector<float> variances,
                               float offset)
      : TestCase(place, alias) {
    input_dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
    anchor_sizes_ = anchor_sizes;
    aspect_ratios_ = aspect_ratios;
    stride_ = stride;
    variances_ = variances;
    offset_ = offset;
  }

  void RunBaseline(Scope* scope) override {
    auto* anchors = scope->NewTensor(anchors_str_);
    auto* vars = scope->NewTensor(variances_str_);
    CHECK(anchors);
    CHECK(vars);

    int num_anchors = anchor_sizes_.size() * aspect_ratios_.size();
    std::vector<int64_t> output_shape(
        {input_dims_[2], input_dims_[3], num_anchors, 4});
    DDim output_dims(output_shape);
    anchors->Resize(output_dims);
    vars->Resize(output_dims);
    auto* anchors_data = anchors->mutable_data<float>();
    auto* vars_data = vars->mutable_data<float>();

    int feature_height = input_dims_[2];
    int feature_width = input_dims_[3];
    float stride_width = stride_[0];
    float stride_height = stride_[1];
    for (int h_idx = 0; h_idx < feature_height; ++h_idx) {
      for (int w_idx = 0; w_idx < feature_width; ++w_idx) {
        float x_ctr = (w_idx * stride_width) + offset_ * (stride_width - 1);
        float y_ctr = (h_idx * stride_height) + offset_ * (stride_height - 1);
        float area, area_ratios;
        float base_w, base_h;
        float scale_w, scale_h;
        float anchor_width, anchor_height;
        auto* anchors_data_hw = anchors_data +
                                h_idx * feature_width * num_anchors * 4 +
                                w_idx * num_anchors * 4;
        for (size_t r = 0; r < aspect_ratios_.size(); ++r) {
          auto ar = aspect_ratios_[r];
          auto* anchors_data_r = anchors_data_hw + r * anchor_sizes_.size() * 4;
          for (size_t s = 0; s < anchor_sizes_.size(); ++s) {
            auto anchor_size = anchor_sizes_[s];
            area = stride_width * stride_height;
            area_ratios = area / ar;
            base_w = round(sqrt(area_ratios));
            base_h = round(base_w * ar);
            scale_w = anchor_size / stride_width;
            scale_h = anchor_size / stride_height;
            anchor_width = scale_w * base_w;
            anchor_height = scale_h * base_h;
            anchors_data_r[s * 4 + 0] = (x_ctr - 0.5 * (anchor_width - 1));
            anchors_data_r[s * 4 + 1] = (y_ctr - 0.5 * (anchor_height - 1));
            anchors_data_r[s * 4 + 2] = (x_ctr + 0.5 * (anchor_width - 1));
            anchors_data_r[s * 4 + 3] = (y_ctr + 0.5 * (anchor_height - 1));
          }
        }
      }
    }

    for (int h = 0; h < feature_height; h++) {
      for (int w = 0; w < feature_width; w++) {
        for (int n = 0; n < num_anchors; n++) {
          auto vars_data_i = vars_data + h * feature_width * num_anchors * 4 +
                             w * num_anchors * 4 + n * 4;
          for (int i = 0; i < 4; i++) {
            vars_data_i[i] = variances_[i];
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("anchor_generator");
    op_desc->SetInput("Input", {input_str_});
    op_desc->SetAttr("anchor_sizes", anchor_sizes_);
    op_desc->SetAttr("aspect_ratios", aspect_ratios_);
    op_desc->SetAttr("stride", stride_);
    op_desc->SetAttr("variances", variances_);
    op_desc->SetAttr("offset", offset_);
    op_desc->SetOutput("Anchors", {anchors_str_});
    op_desc->SetOutput("Variances", {variances_str_});
  }

  void PrepareData() override {
    std::vector<float> input_data(input_dims_.production());
    for (int i = 0; i < input_dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      input_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(input_str_, input_dims_, input_data.data());
  }
};

TEST(AnchorGenerator, precision) {
  Place place;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  for (int n : {1, 3}) {
    for (int c : {3, 6}) {
      for (int h : {9, 18}) {
        for (int w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(
              new AnchorGeneratorComputeTester(place,
                                               "def",
                                               n,
                                               c,
                                               h,
                                               w,
                                               {64, 128, 256, 512},
                                               {0.5, 1.0, 2.0},
                                               {16.0, 16.0},
                                               {0.1, 0.1, 0.2, 0.2},
                                               0.5));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
