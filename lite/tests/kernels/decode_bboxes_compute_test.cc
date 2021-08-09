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

class DecodeBboxesComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string loc_ = "loc";
  std::string prior_ = "prior";
  std::string bbox_ = "bbox";
  int batch_num_;
  int num_priors_;
  int num_loc_classes_{0};
  int background_label_id_{0};
  bool share_location_{true};
  bool variance_encoded_in_target_;
  std::string code_type_;
  DDim loc_dims_;
  DDim prior_dims_;

 public:
  DecodeBboxesComputeTester(const Place& place,
                            const std::string& alias,
                            int batch_num,
                            int num_priors,
                            int num_loc_classes,
                            int background_label_id,
                            bool share_location,
                            bool variance_encoded_in_target,
                            std::string code_type,
                            DDim loc_dims,
                            DDim prior_dims)
      : TestCase(place, alias),
        batch_num_(batch_num),
        num_priors_(num_priors),
        num_loc_classes_(num_loc_classes),
        background_label_id_(background_label_id),
        share_location_(share_location),
        variance_encoded_in_target_(variance_encoded_in_target),
        code_type_(code_type),
        loc_dims_(loc_dims),
        prior_dims_(prior_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* bbox = scope->NewTensor(bbox_);
    CHECK(bbox);
    bbox->Resize(loc_dims_);
    auto* bbox_data = bbox->mutable_data<float>();

    auto* loc = scope->FindTensor(loc_);
    const auto* loc_data = loc->data<float>();

    auto* prior = scope->FindTensor(prior_);
    const auto* prior_data = prior->data<float>();

    for (int n = 0; n < batch_num_; ++n) {
      const float* ptr_loc_batch = loc_data + n * num_priors_ * 4;
      float* ptr_bbox_batch = bbox_data + n * num_priors_ * 4;
      for (int i = 0; i < num_priors_; ++i) {
        int idx = i * 4;
        const float* ptr_loc = ptr_loc_batch + idx;
        const float* ptr_prior = prior_data + idx;
        float* ptr_bbox = ptr_bbox_batch + idx;

        float p_xmin = ptr_prior[0];
        float p_ymin = ptr_prior[1];
        float p_xmax = ptr_prior[2];
        float p_ymax = ptr_prior[3];
        float prior_width = p_xmax - p_xmin;
        float prior_height = p_ymax - p_ymin;
        float prior_center_x = (p_xmin + p_xmax) / 2.f;
        float prior_center_y = (p_ymin + p_ymax) / 2.f;

        float xmin = ptr_loc[0];
        float ymin = ptr_loc[1];
        float xmax = ptr_loc[2];
        float ymax = ptr_loc[3];

        if (code_type_ == "corner") {
          if (variance_encoded_in_target_) {
            ptr_bbox[0] = ptr_loc[0] + ptr_prior[0];
            ptr_bbox[1] = ptr_loc[1] + ptr_prior[1];
            ptr_bbox[2] = ptr_loc[2] + ptr_prior[2];
            ptr_bbox[3] = ptr_loc[3] + ptr_prior[3];
          } else {
            const float* variance_data = prior_data + 4 * num_priors_;
            const float* ptr_var = variance_data + idx;
            ptr_bbox[0] = ptr_var[0] * ptr_loc[0] + ptr_prior[0];
            ptr_bbox[1] = ptr_var[1] * ptr_loc[1] + ptr_prior[1];
            ptr_bbox[2] = ptr_var[2] * ptr_loc[2] + ptr_prior[2];
            ptr_bbox[3] = ptr_var[3] * ptr_loc[3] + ptr_prior[3];
          }
        } else if (code_type_ == "center_size") {
          float decode_bbox_center_x;
          float decode_bbox_center_y;
          float decode_bbox_width;
          float decode_bbox_height;
          if (variance_encoded_in_target_) {
            //! variance is encoded in target, we simply need to retore the
            //! offset
            //! predictions.
            decode_bbox_center_x = xmin * prior_width + prior_center_x;
            decode_bbox_center_y = ymin * prior_height + prior_center_y;
            decode_bbox_width = std::exp(xmax) * prior_width;
            decode_bbox_height = std::exp(ymax) * prior_height;
          } else {
            const float* variance_data = prior_data + 4 * num_priors_;
            const float* ptr_var = variance_data + idx;
            decode_bbox_center_x =
                ptr_var[0] * xmin * prior_width + prior_center_x;
            decode_bbox_center_y =
                ptr_var[1] * ymin * prior_height + prior_center_y;
            decode_bbox_width = std::exp(ptr_var[2] * xmax) * prior_width;
            decode_bbox_height = std::exp(ptr_var[3] * ymax) * prior_height;
          }
          ptr_bbox[0] = decode_bbox_center_x - decode_bbox_width / 2.f;
          ptr_bbox[1] = decode_bbox_center_y - decode_bbox_height / 2.f;
          ptr_bbox[2] = decode_bbox_center_x + decode_bbox_width / 2.f;
          ptr_bbox[3] = decode_bbox_center_y + decode_bbox_height / 2.f;
        } else if (code_type_ == "corner_size") {
          if (variance_encoded_in_target_) {
            ptr_bbox[0] = p_xmin + ptr_loc[0] * prior_width;
            ptr_bbox[1] = p_ymin + ptr_loc[1] * prior_height;
            ptr_bbox[2] = p_xmax + ptr_loc[2] * prior_width;
            ptr_bbox[3] = p_ymax + ptr_loc[3] * prior_height;
          } else {
            const float* variance_data = prior_data + 4 * num_priors_;
            const float* ptr_var = variance_data + idx;
            ptr_bbox[0] = p_xmin + ptr_loc[0] * ptr_var[0] * prior_width;
            ptr_bbox[1] = p_ymin + ptr_loc[1] * ptr_var[1] * prior_width;
            ptr_bbox[2] = p_xmax + ptr_loc[2] * ptr_var[2] * prior_width;
            ptr_bbox[3] = p_ymax + ptr_loc[3] * ptr_var[3] * prior_width;
          }
        } else {
          LOG(FATAL) << "unsupported code type: " << code_type_;
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("decode_bboxes");
    op_desc->SetInput("Loc", {loc_});
    op_desc->SetInput("Prior", {prior_});
    op_desc->SetOutput("Bbox", {bbox_});
    op_desc->SetAttr("batch_num", batch_num_);
    op_desc->SetAttr("num_priors", num_priors_);
    op_desc->SetAttr("num_loc_classes", num_loc_classes_);
    op_desc->SetAttr("background_label_id", background_label_id_);
    op_desc->SetAttr("share_location", share_location_);
    op_desc->SetAttr("variance_encoded_in_target", variance_encoded_in_target_);
    op_desc->SetAttr("code_type", code_type_);
  }

  void PrepareData() override {
    std::vector<float> loc_data(loc_dims_.production());
    std::vector<float> prior_data(prior_dims_.production());

    for (int i = 0; i < loc_dims_.production(); i++) {
      loc_data[i] = i * 1.1 / loc_dims_.production();
    }
    for (int i = 0; i < prior_dims_.production(); i++) {
      prior_data[i] = i * 1.2 / prior_dims_.production();
    }

    SetCommonTensor(loc_, loc_dims_, loc_data.data());
    SetCommonTensor(prior_, prior_dims_, prior_data.data());
  }
};

void test_decode_bboxes(Place place) {
  for (int batch_num : {1, 2, 3, 4}) {
    for (int num_priors : {1, 3, 4, 8, 10}) {
      for (std::string code_type : {"corner", "center_size", "corner_size"}) {
        for (bool variance_encoded_in_target : {true, false}) {
          std::unique_ptr<arena::TestCase> tester(
              new DecodeBboxesComputeTester(place,
                                            "def",
                                            batch_num,
                                            num_priors,
                                            0,
                                            0,
                                            true,
                                            variance_encoded_in_target,
                                            code_type,
                                            DDim({batch_num, num_priors * 4}),
                                            DDim({1, 2, num_priors * 4})));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(DecodeBboxes, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_decode_bboxes(place);
#endif
}

}  // namespace lite
}  // namespace paddle
