// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/density_prior_box_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/core/op_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void inferShape_(Tensor* input,
                 Tensor* boxes,
                 Tensor* variances,
                 std::vector<float> fixed_ratios,
                 std::vector<int> densities) {
  auto feat_height = input->dims()[2];
  auto feat_width = input->dims()[3];

  int num_priors = 0;
  for (size_t i = 0; i < densities.size(); ++i) {
    num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
  }

  std::vector<int64_t> boxes_shape = {feat_width, feat_height, num_priors, 4};
  std::vector<int64_t> vars_shape = boxes_shape;
  boxes->Resize(boxes_shape);
  variances->Resize(vars_shape);
}

void prior_density_box_ref(
    const std::shared_ptr<operators::DensityPriorBoxOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto input =
      scope->FindVar(op_info->Input("Input").front())->GetMutable<Tensor>();
  auto image =
      scope->FindVar(op_info->Input("Image").front())->GetMutable<Tensor>();
  auto boxes_tensor =
      scope->FindVar(op_info->Output("Boxes").front())->GetMutable<Tensor>();
  auto variances = scope->FindVar(op_info->Output("Variances").front())
                       ->GetMutable<Tensor>();
  auto clip = op_info->GetAttr<bool>("clip");
  auto fixed_sizes = op_info->GetAttr<std::vector<float>>("fixed_sizes");
  auto fixed_ratios = op_info->GetAttr<std::vector<float>>("fixed_ratios");
  auto variances_ = op_info->GetAttr<std::vector<float>>("variances");
  auto densities = op_info->GetAttr<std::vector<int>>("densities");
  auto offset = op_info->GetAttr<float>("offset");
  auto step_w = op_info->GetAttr<float>("step_w");
  auto step_h = op_info->GetAttr<float>("step_h");

  std::vector<int> input_shape = {128, 128};
  std::vector<int> image_shape = {256, 256};
  int num_priors = 0;
  for (size_t i = 0; i < densities.size(); ++i) {
    num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
  }

  int boxes_count = boxes_tensor->dims().production();

  float* boxes = boxes_tensor->mutable_data<float>();
  float* vars = variances->mutable_data<float>();

  auto img_width = image->dims()[3];
  auto img_height = image->dims()[2];

  auto feature_width = input->dims()[3];
  auto feature_height = input->dims()[2];

  float step_width, step_height;
  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = step_w;
    step_height = step_h;
  }

  int step_average = static_cast<int>((step_width + step_height) * 0.5);

  std::vector<float> sqrt_fixed_ratios;
  for (size_t i = 0; i < fixed_ratios.size(); i++) {
    sqrt_fixed_ratios.push_back(sqrt(fixed_ratios[i]));
  }

  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      float center_x = (w + offset) * step_width;
      float center_y = (h + offset) * step_height;
      int idx = 0;
      // Generate density prior boxes with fixed sizes.
      for (size_t s = 0; s < fixed_sizes.size(); ++s) {
        auto fixed_size = fixed_sizes[s];
        int density = densities[s];
        int shift = step_average / density;
        // Generate density prior boxes with fixed ratios.
        for (size_t r = 0; r < fixed_ratios.size(); ++r) {
          float box_width_ratio = fixed_size * sqrt_fixed_ratios[r];
          float box_height_ratio = fixed_size / sqrt_fixed_ratios[r];
          float density_center_x = center_x - step_average / 2. + shift / 2.;
          float density_center_y = center_y - step_average / 2. + shift / 2.;
          for (int di = 0; di < density; ++di) {
            for (int dj = 0; dj < density; ++dj) {
              float center_x_temp = density_center_x + dj * shift;
              float center_y_temp = density_center_y + di * shift;
              boxes[h * feature_width * num_priors * 4 + w * num_priors * 4 +
                    idx * 4 + 0] =
                  std::max((center_x_temp - box_width_ratio / 2.) / img_width,
                           0.);
              boxes[h * feature_width * num_priors * 4 + w * num_priors * 4 +
                    idx * 4 + 1] =
                  std::max((center_y_temp - box_height_ratio / 2.) / img_height,
                           0.);
              boxes[h * feature_width * num_priors * 4 + w * num_priors * 4 +
                    idx * 4 + 2] =
                  std::min((center_x_temp + box_width_ratio / 2.) / img_width,
                           1.);
              boxes[h * feature_width * num_priors * 4 + w * num_priors * 4 +
                    idx * 4 + 3] =
                  std::min((center_y_temp + box_height_ratio / 2.) / img_height,
                           1.);
              idx++;
            }
          }
        }
      }
    }
  }
  if (clip) {
    std::transform(boxes, boxes + boxes_count, boxes, [](float v) -> float {
      return std::min<float>(std::max<float>(v, 0.), 1.);
    });
  }
  int box_num = feature_height * feature_width * num_priors;

  for (int i = 0; i < box_num; ++i) {
    for (size_t j = 0; j < variances_.size(); ++j) {
      vars[i * variances_.size() + j] = variances_[j];
    }
  }
}

void test_prior_density_box(int feat_h,
                            int feat_w,
                            int img_h,
                            int img_w,
                            bool clip,
                            std::vector<float> fixed_sizes,
                            std::vector<float> fixed_ratios,
                            std::vector<float> variances_,
                            std::vector<int> densities,
                            float step_w,
                            float step_h,
                            float offset) {
  // prepare input&output variables
  Scope scope;
  std::string input_var_name("Input");
  std::string image_var_name("Image");
  std::string boxes_var_name("Boxes");
  std::string variances_var_name("Variances");
  std::string boxes_ref_var_name("Boxes_ref");
  std::string variances_ref_var_name("Variances_ref");
  auto* input = scope.Var(input_var_name)->GetMutable<Tensor>();
  auto* image = scope.Var(image_var_name)->GetMutable<Tensor>();
  auto* boxes = scope.Var(boxes_var_name)->GetMutable<Tensor>();
  auto* variances = scope.Var(variances_var_name)->GetMutable<Tensor>();
  auto* boxes_ref = scope.Var(boxes_ref_var_name)->GetMutable<Tensor>();
  auto* variances_ref = scope.Var(variances_ref_var_name)->GetMutable<Tensor>();
  input->Resize({1, 1, feat_h, feat_w});
  image->Resize({1, 1, img_h, img_w});

  // initialize input&output data
  FillTensor<float>(input);
  FillTensor<float, int>(image);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("density_prior_box");
  opdesc.SetInput("Input", {input_var_name});
  opdesc.SetInput("Image", {image_var_name});
  opdesc.SetOutput("Boxes", {boxes_var_name});
  opdesc.SetOutput("Variances", {variances_var_name});

  opdesc.SetAttr("fixed_sizes", fixed_sizes);
  opdesc.SetAttr("fixed_ratios", fixed_ratios);
  opdesc.SetAttr("variances", variances_);
  opdesc.SetAttr("densities", densities);
  opdesc.SetAttr("offset", offset);
  opdesc.SetAttr("clip", clip);
  opdesc.SetAttr("step_w", step_w);
  opdesc.SetAttr("step_h", step_h);

  inferShape_(input, boxes, variances, fixed_ratios, densities);
  inferShape_(input, boxes_ref, variances_ref, fixed_ratios, densities);

  auto op = CreateOp<operators::DensityPriorBoxOpLite>(opdesc, &scope);
  prior_density_box_ref(op);
  boxes_ref->CopyDataFrom(*boxes);
  variances_ref->CopyDataFrom(*variances);
  LaunchOp(op,
           {input_var_name, image_var_name},
           {boxes_var_name, variances_var_name});

  // execute reference implementation and save to output tensor('out')

  // ===================== Trans From NHWC to NCHW ====================
  Tensor boxes_trans;
  boxes_trans.Resize(boxes->dims().Vectorize());
  transpose(boxes->mutable_data<float>(),
            boxes_trans.mutable_data<float>(),
            {static_cast<int>(boxes->dims()[0]),
             static_cast<int>(boxes->dims()[2]),
             static_cast<int>(boxes->dims()[3]),
             static_cast<int>(boxes->dims()[1])},
            {0, 3, 1, 2});
  boxes->CopyDataFrom(boxes_trans);
  Tensor vars_trans;
  vars_trans.Resize(variances->dims().Vectorize());
  transpose(variances->mutable_data<float>(),
            vars_trans.mutable_data<float>(),
            {static_cast<int>(variances->dims()[0]),
             static_cast<int>(variances->dims()[2]),
             static_cast<int>(variances->dims()[3]),
             static_cast<int>(variances->dims()[1])},
            {0, 3, 1, 2});
  variances->CopyDataFrom(vars_trans);

  // compare results
  auto* boxes_data = boxes->mutable_data<float>();
  auto* boxes_ref_data = boxes_ref->mutable_data<float>();
  auto* variances_data = variances->mutable_data<float>();
  auto* variances_ref_data = variances_ref->mutable_data<float>();

  // ToFile(*variances, "var_mlu.txt");
  // ToFile(*variances_ref, "var_cpu.txt");
  // ToFile(*boxes, "box_mlu.txt");
  // ToFile(*boxes_ref, "box_cpu.txt");
  for (int i = 0; i < variances->dims().production(); i++) {
    VLOG(6) << i;
    EXPECT_NEAR(variances_data[i], variances_ref_data[i], 1e-5);
  }

  for (int i = 0; i < boxes->dims().production(); i++) {
    VLOG(6) << i;
    EXPECT_NEAR(boxes_data[i], boxes_ref_data[i], 1e-5);
  }
}

TEST(MLUBridges, prior_density_box) {
  // std::vector<int> input_shape = {128, 128};
  // std::vector<int> image_shape = {256, 256};
  // std::vector<float> fixed_sizes = {8 * 16, 16 * 16, 32 * 16};
  // std::vector<float> fixed_sizes = {8, 16, 32};
  // std::vector<float> fixed_ratios = {0.5, 1, 2};
  // std::vector<int> densities = {1, 1, 1};

  std::vector<int> input_shape = {16, 16};
  std::vector<int> image_shape = {32, 32};
  std::vector<float> fixed_sizes = {8, 16, 32};
  std::vector<float> fixed_ratios = {0.5, 1, 2};
  std::vector<int> densities = {1, 1, 1};
  std::vector<float> variances = {0.1, 0.1, 0.2, 0.2};
  bool clip = true;
  float offset = 0.5;
  float step_h = 0;
  float step_w = 0;

  test_prior_density_box(input_shape[1],
                         input_shape[0],
                         image_shape[1],
                         image_shape[0],
                         clip,
                         fixed_sizes,
                         fixed_ratios,
                         variances,
                         densities,
                         offset,
                         step_h,
                         step_w);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(density_prior_box, kMLU);
