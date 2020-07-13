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

#include <cmath>
#include <string>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/decode_bboxes_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void decode_bboxes_compute_ref(const operators::DecodeBboxesParam& param) {
  const dtype* loc_data = param.loc_data->data<const dtype>();
  const dtype* prior_data = param.prior_data->data<const dtype>();
  dtype* bbox_data = param.bbox_data->mutable_data<dtype>();

  for (int n = 0; n < param.batch_num; ++n) {
    const dtype* ptr_loc_batch = loc_data + n * param.num_priors * 4;
    dtype* ptr_bbox_batch = bbox_data + n * param.num_priors * 4;
    for (int i = 0; i < param.num_priors; ++i) {
      int idx = i * 4;
      const dtype* ptr_loc = ptr_loc_batch + idx;
      const dtype* ptr_prior = prior_data + idx;
      dtype* ptr_bbox = ptr_bbox_batch + idx;

      dtype p_xmin = ptr_prior[0];
      dtype p_ymin = ptr_prior[1];
      dtype p_xmax = ptr_prior[2];
      dtype p_ymax = ptr_prior[3];
      dtype prior_width = p_xmax - p_xmin;
      dtype prior_height = p_ymax - p_ymin;
      dtype prior_center_x = (p_xmin + p_xmax) / 2.f;
      dtype prior_center_y = (p_ymin + p_ymax) / 2.f;

      dtype xmin = ptr_loc[0];
      dtype ymin = ptr_loc[1];
      dtype xmax = ptr_loc[2];
      dtype ymax = ptr_loc[3];

      if (param.code_type == "corner") {
        if (param.variance_encoded_in_target) {
          ptr_bbox[0] = ptr_loc[0] + ptr_prior[0];
          ptr_bbox[1] = ptr_loc[1] + ptr_prior[1];
          ptr_bbox[2] = ptr_loc[2] + ptr_prior[2];
          ptr_bbox[3] = ptr_loc[3] + ptr_prior[3];
        } else {
          const dtype* variance_data = prior_data + 4 * param.num_priors;
          const dtype* ptr_var = variance_data + idx;
          ptr_bbox[0] = ptr_var[0] * ptr_loc[0] + ptr_prior[0];
          ptr_bbox[1] = ptr_var[1] * ptr_loc[1] + ptr_prior[1];
          ptr_bbox[2] = ptr_var[2] * ptr_loc[2] + ptr_prior[2];
          ptr_bbox[3] = ptr_var[3] * ptr_loc[3] + ptr_prior[3];
        }
      } else if (param.code_type == "center_size") {
        dtype decode_bbox_center_x;
        dtype decode_bbox_center_y;
        dtype decode_bbox_width;
        dtype decode_bbox_height;
        if (param.variance_encoded_in_target) {
          //! variance is encoded in target, we simply need to retore the offset
          //! predictions.
          decode_bbox_center_x = xmin * prior_width + prior_center_x;
          decode_bbox_center_y = ymin * prior_height + prior_center_y;
          decode_bbox_width = std::exp(xmax) * prior_width;
          decode_bbox_height = std::exp(ymax) * prior_height;
        } else {
          const dtype* variance_data = prior_data + 4 * param.num_priors;
          const dtype* ptr_var = variance_data + idx;
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
      } else if (param.code_type == "corner_size") {
        if (param.variance_encoded_in_target) {
          ptr_bbox[0] = p_xmin + ptr_loc[0] * prior_width;
          ptr_bbox[1] = p_ymin + ptr_loc[1] * prior_height;
          ptr_bbox[2] = p_xmax + ptr_loc[2] * prior_width;
          ptr_bbox[3] = p_ymax + ptr_loc[3] * prior_height;
        } else {
          const dtype* variance_data = prior_data + 4 * param.num_priors;
          const dtype* ptr_var = variance_data + idx;
          ptr_bbox[0] = p_xmin + ptr_loc[0] * ptr_var[0] * prior_width;
          ptr_bbox[1] = p_ymin + ptr_loc[1] * ptr_var[1] * prior_width;
          ptr_bbox[2] = p_xmax + ptr_loc[2] * ptr_var[2] * prior_width;
          ptr_bbox[3] = p_ymax + ptr_loc[3] * ptr_var[3] * prior_width;
        }
      } else {
        LOG(FATAL) << "unsupported code type: " << param.code_type;
      }
    }
  }
}

TEST(decode_bboxes_arm, retrive_op) {
  auto decode_bboxes = KernelRegistry::Global().Create("decode_bboxes");
  ASSERT_FALSE(decode_bboxes.empty());
  ASSERT_TRUE(decode_bboxes.front());
}

TEST(decode_bboxes_arm, init) {
  DecodeBboxesCompute decode_bboxes;
  ASSERT_EQ(decode_bboxes.precision(), PRECISION(kFloat));
  ASSERT_EQ(decode_bboxes.target(), TARGET(kARM));
}

TEST(decode_bboxes_arm, compute) {
  DecodeBboxesCompute decode_bboxes;
  operators::DecodeBboxesParam param;
  lite::Tensor loc, prior, bbox, bbox_ref;

  for (int batch_num : {1, 2, 3, 4}) {
    for (int num_priors : {1, 3, 4, 8, 10}) {
      for (std::string code_type : {"corner", "center_size", "corner_size"}) {
        for (bool variance_encoded_in_target : {true, false}) {
          auto loc_dim =
              DDim(std::vector<int64_t>({batch_num, num_priors * 4}));
          loc.Resize(loc_dim);
          auto prior_dim = DDim(std::vector<int64_t>({1, 2, num_priors * 4}));
          prior.Resize(prior_dim);
          bbox.Resize(loc_dim);
          bbox_ref.Resize(loc_dim);
          auto* loc_data = loc.mutable_data<float>();
          auto* prior_data = prior.mutable_data<float>();
          auto* bbox_data = bbox.mutable_data<float>();
          auto* bbox_ref_data = bbox_ref.mutable_data<float>();

          for (int i = 0; i < loc_dim.production(); ++i) {
            loc_data[i] = i * 1. / loc_dim.production();
          }
          for (int i = 0; i < prior_dim.production(); ++i) {
            prior_data[i] = i * 1. / prior_dim.production();
          }

          param.loc_data = &loc;
          param.prior_data = &prior;
          param.bbox_data = &bbox;
          param.num_loc_classes = 0;
          param.share_location = true;
          param.batch_num = batch_num;
          param.num_priors = num_priors;
          param.code_type = code_type;
          param.variance_encoded_in_target = variance_encoded_in_target;
          decode_bboxes.SetParam(param);
          decode_bboxes.Run();
          param.bbox_data = &bbox_ref;
          decode_bboxes_compute_ref<float>(param);
          for (int i = 0; i < bbox.dims().production(); i++) {
            EXPECT_NEAR(bbox_data[i], bbox_ref_data[i], 1e-5);
          }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(decode_bboxes, kARM, kFloat, kNCHW, def);
