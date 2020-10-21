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

static inline void box_coder_ref(lite::Tensor* proposals,
                                 const lite::Tensor* anchors,
                                 const lite::Tensor* bbox_deltas,
                                 const lite::Tensor* variances,
                                 int axis,
                                 bool box_normalized,
                                 std::string code_type) {
  if (code_type == "decode_center_size") {
    const size_t row = bbox_deltas->dims()[0];
    const size_t col = bbox_deltas->dims()[1];
    int anchor_len = 4;
    int out_len = 4;
    int var_len = 4;
    int delta_len = 4;
    const float* anchor_data = anchors->data<float>();
    const float* bbox_deltas_data = bbox_deltas->data<float>();
    float* proposals_data = proposals->mutable_data<float>();
    const float* variances_data = variances->data<float>();
    float normalized = !box_normalized ? 1.f : 0;

    for (int64_t row_id = 0; row_id < row; ++row_id) {
      for (int64_t col_id = 0; col_id < col; ++col_id) {
        size_t delta_offset = row_id * col * delta_len + col_id * delta_len;
        size_t out_offset = row_id * col * out_len + col_id * out_len;
        int prior_box_offset =
            axis == 0 ? col_id * anchor_len : row_id * anchor_len;
        int var_offset = axis == 0 ? col_id * var_len : row_id * var_len;
        auto anchor_data_tmp = anchor_data + prior_box_offset;
        auto bbox_deltas_data_tmp = bbox_deltas_data + delta_offset;
        auto proposals_data_tmp = proposals_data + out_offset;
        auto anchor_width =
            anchor_data_tmp[2] - anchor_data_tmp[0] + normalized;
        auto anchor_height =
            anchor_data_tmp[3] - anchor_data_tmp[1] + normalized;
        auto anchor_center_x = anchor_data_tmp[0] + 0.5 * anchor_width;
        auto anchor_center_y = anchor_data_tmp[1] + 0.5 * anchor_height;
        float bbox_center_x = 0, bbox_center_y = 0;
        float bbox_width = 0, bbox_height = 0;

        auto variances_data_tmp = variances_data + var_offset;
        bbox_center_x =
            variances_data_tmp[0] * bbox_deltas_data_tmp[0] * anchor_width +
            anchor_center_x;
        bbox_center_y =
            variances_data_tmp[1] * bbox_deltas_data_tmp[1] * anchor_height +
            anchor_center_y;
        bbox_width = std::exp(variances_data_tmp[2] * bbox_deltas_data_tmp[2]) *
                     anchor_width;
        bbox_height =
            std::exp(variances_data_tmp[3] * bbox_deltas_data_tmp[3]) *
            anchor_height;

        proposals_data_tmp[0] = bbox_center_x - bbox_width / 2;
        proposals_data_tmp[1] = bbox_center_y - bbox_height / 2;
        proposals_data_tmp[2] = bbox_center_x + bbox_width / 2 - normalized;
        proposals_data_tmp[3] = bbox_center_y + bbox_height / 2 - normalized;
      }
    }
  } else if (code_type == "encode_center_size") {
    LOG(FATAL) << "not implemented type: " << code_type;
  } else {
    LOG(FATAL) << "not supported type: " << code_type;
  }
}

class BoxCoderComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string prior_box_ = "PriorBox";
  std::string prior_box_var_ = "PriorBoxVar";
  std::string target_box_ = "TargetBox";
  std::string output_box_ = "OutputBox";

  int axis_;
  bool box_normalized_{true};
  std::string code_type_;
  DDim prior_box_dims_;
  DDim prior_box_var_dims_;
  DDim target_box_dims_;

 public:
  BoxCoderComputeTester(const Place& place,
                        const std::string& alias,
                        int axis,
                        bool box_normalized,
                        std::string code_type,
                        DDim prior_box_dims,
                        DDim prior_box_var_dims,
                        DDim target_box_dims)
      : TestCase(place, alias),
        axis_(axis),
        box_normalized_(box_normalized),
        code_type_(code_type),
        prior_box_dims_(prior_box_dims),
        prior_box_var_dims_(prior_box_var_dims),
        target_box_dims_(target_box_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* output_box = scope->NewTensor(output_box_);
    CHECK(output_box);
    output_box->Resize(target_box_dims_);

    auto* prior_box = scope->FindTensor(prior_box_);
    auto* prior_box_var = scope->FindTensor(prior_box_var_);
    auto* target_box = scope->FindTensor(target_box_);

    box_coder_ref(output_box,
                  prior_box,
                  target_box,
                  prior_box_var,
                  axis_,
                  box_normalized_,
                  code_type_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("box_coder");
    op_desc->SetInput("PriorBox", {prior_box_});
    op_desc->SetInput("PriorBoxVar", {prior_box_var_});
    op_desc->SetInput("TargetBox", {target_box_});
    op_desc->SetOutput("OutputBox", {output_box_});

    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("box_normalized", box_normalized_);
    op_desc->SetAttr("code_type", code_type_);
  }

  void PrepareData() override {
    std::vector<float> prior_box_data(prior_box_dims_.production());
    std::vector<float> prior_box_var_data(prior_box_var_dims_.production());
    std::vector<float> target_box_data(target_box_dims_.production());

    for (int i = 0; i < prior_box_dims_.production(); i++) {
      prior_box_data[i] = i * 1.1 / prior_box_dims_.production();
    }
    for (int i = 0; i < prior_box_var_dims_.production(); i++) {
      prior_box_var_data[i] = i * 1.2 / prior_box_var_dims_.production();
    }
    for (int i = 0; i < target_box_dims_.production(); i++) {
      target_box_data[i] = i * 1.3 / target_box_dims_.production();
    }

    SetCommonTensor(prior_box_, prior_box_dims_, prior_box_data.data());
    SetCommonTensor(
        prior_box_var_, prior_box_var_dims_, prior_box_var_data.data());
    SetCommonTensor(target_box_, target_box_dims_, target_box_data.data());
  }
};

void test_box_coder(Place place) {
  for (int N : {1, 2, 3, 4}) {
    for (int M : {1, 3, 4, 8, 10}) {
      int axis = 0;
      for (bool norm : {true, false}) {
        for (std::string code_type : {"decode_center_size"}) {
          std::unique_ptr<arena::TestCase> tester(
              new BoxCoderComputeTester(place,
                                        "def",
                                        axis,
                                        norm,
                                        code_type,
                                        DDim({M, 4}),
                                        DDim({M, 4}),
                                        DDim({N, M, 4})));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(BoxCoder, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  test_box_coder(place);
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_box_coder(place);
#endif
}

}  // namespace lite
}  // namespace paddle
