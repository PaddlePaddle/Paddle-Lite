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
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

void nearest_interp(const float* src,
                    int w_in,
                    int h_in,
                    float* dst,
                    int w_out,
                    int h_out,
                    float scale_x,
                    float scale_y,
                    bool align_corners) {
  float scale_w_new;
  float scale_h_new;
  if (align_corners) {
    scale_w_new = static_cast<float>(w_in - 1) / (w_out - 1);
    scale_h_new = static_cast<float>(h_in - 1) / (h_out - 1);
  } else {
    scale_w_new = static_cast<float>(w_in / w_out);
    scale_h_new = static_cast<float>(h_in / h_out);
  }

#pragma omp parallel for collapse(2) schedule(static)
  for (int h = 0; h < h_out; ++h) {
    for (int w = 0; w < w_out; ++w) {
      int near_x = static_cast<int>(scale_w_new * w + 0.5);
      int near_y = static_cast<int>(scale_h_new * h + 0.5);
      near_x = near_x < 0 ? 0 : near_x;
      near_y = near_y < 0 ? 0 : near_y;
      dst[h * w_out + w] = src[near_y * w_in + near_x];
    }
  }
}

class NearestInterpComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string input1_ = "OutSize";
  std::string output_ = "Out";

  float height_scale_ = 0.0f;
  float width_scale_ = 0.0f;
  int out_height_ = -1;
  int out_width_ = -1;
  bool align_corners_ = true;
  std::string interp_method_ = "Nearest";
  DDim dims_{{2, 3}};
  DDim _dims0_{{2, 3, 3, 2}};
  DDim _dims1_{{2}};

 public:
  NearestInterpComputeTester(const Place& place,
                             const std::string& alias,
                             float height_scale,
                             float width_scale,
                             int out_height,
                             int out_width,
                             bool align_corners,
                             std::string interp_method)
      : TestCase(place, alias),
        height_scale_(height_scale),
        width_scale_(width_scale),
        out_height_(out_height),
        out_width_(out_width),
        align_corners_(align_corners),
        interp_method_(interp_method) {}

  void RunBaseline(Scope* scope) override {
    width_scale_ = height_scale_;
    auto* outputs = scope->NewTensor(output_);
    CHECK(outputs);
    outputs->Resize(dims_);
    std::vector<const lite::Tensor*> inputs;
    inputs.emplace_back(scope->FindTensor(input0_));
    inputs.emplace_back(scope->FindTensor(input1_));

    auto outsize_data = inputs[1]->data<int>();
    if (out_width_ != -1 && out_height_ != -1) {
      height_scale_ = static_cast<float>(out_height_ / inputs[0]->dims()[2]);
      width_scale_ = static_cast<float>(out_width_ / inputs[0]->dims()[3]);
    }
    if (inputs.size() > 1) {
      int h_out = outsize_data[0];  // HW
      int w_out = outsize_data[1];  // HW
      int num_cout = outputs->dims()[0];
      int c_cout = outputs->dims()[1];
      outputs->Resize({num_cout, c_cout, h_out, w_out});
    }
    float* dout = outputs->mutable_data<float>();
    const float* din = inputs[0]->data<float>();
    int out_num = outputs->dims()[0];
    int out_c = outputs->dims()[1];
    int count = out_num * out_c;
    int in_h = inputs[0]->dims()[2];
    int in_w = inputs[0]->dims()[3];
    int out_h = outputs->dims()[2];
    int out_w = outputs->dims()[3];
    int spatial_in = in_h * in_w;
    int spatial_out = out_h * out_w;
    nearest_interp(din,
                   in_w,
                   in_h,
                   dout,
                   out_w,
                   out_h,
                   1.f / width_scale_,
                   1.f / height_scale_,
                   align_corners_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("nearest_interp");
    op_desc->SetInput("X", {input0_});
    op_desc->SetInput("OutSize", {input1_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("scale", height_scale_);
    op_desc->SetAttr("out_h", out_height_);
    op_desc->SetAttr("out_w", out_width_);
    op_desc->SetAttr("align_corners", align_corners_);
    op_desc->SetAttr("interp_method", interp_method_);
  }

  void PrepareData() override {
    std::vector<float> data0(_dims0_.production());
    for (int i = 0; i < _dims0_.production(); i++) {
      data0[i] = i * 1.1;
    }

    std::vector<int> data1(_dims1_.production());
    for (int i = 0; i < _dims1_.production(); i++) {
      data1[i] = (i + 1) * 2;
    }

    SetCommonTensor(input0_, _dims0_, data0.data());
    SetCommonTensor(input1_, _dims1_, data1.data());
  }
};

void test_nearest_interp(Place place) {
  std::string interp_method = "Nearest";
  for (float scale : {0.123, 2., 1.2}) {
    for (int out_height : {2, 1, 6}) {
      for (int out_width : {2, 3, 5}) {
        for (bool align_corners : {true, false}) {
          std::unique_ptr<arena::TestCase> tester(
              new NearestInterpComputeTester(place,
                                             "def",
                                             scale,
                                             scale,
                                             out_height,
                                             out_width,
                                             align_corners,
                                             interp_method));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(NearestInterp, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_nearest_interp(place);
#endif
}

}  // namespace lite
}  // namespace paddle
