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

#include <bmcompiler_if.h>
#include <math.h>
#include <user_bmcpu_common.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

typedef struct __tag_st_priorbox_param {
  std::vector<float> fixed_sizes;
  std::vector<float> fixed_ratios;
  std::vector<int> densities;
  std::vector<float> variances;
  float step_w;
  float step_h;
  float offset;
  int prior_num;
  bool clip;
  bool flatten_to_2d;
} st_priorbox_param;

float* compute_density_priorbox_kernel(OpLite* op, st_priorbox_param* param) {
  auto op_info = op->op_info();
  auto scope = op->scope();
  // inputs
  auto in_var_name = op_info->Input("Input").front();
  auto in = scope->FindVar(in_var_name)->GetMutable<lite::Tensor>();
  auto in_dims = in->dims();
  auto img_var_name = op_info->Input("Image").front();
  auto img = scope->FindVar(img_var_name)->GetMutable<lite::Tensor>();
  auto img_dims = img->dims();
  // outputs
  auto boxes_var_name = op_info->Output("Boxes").front();
  auto boxes = scope->FindVar(boxes_var_name)->GetMutable<lite::Tensor>();
  auto var_var_name = op_info->Output("Variances").front();
  auto var = scope->FindVar(var_var_name)->GetMutable<lite::Tensor>();

  auto img_width = img_dims[3];
  auto img_height = img_dims[2];
  auto feature_width = in_dims[3];
  auto feature_height = in_dims[2];
  float step_width, step_height;
  if (param->step_w == 0.f || param->step_h == 0.f) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = param->step_w;
    step_height = param->step_h;
  }
  int num_priors = 0;
  for (size_t i = 0; i < param->densities.size(); ++i) {
    num_priors += (param->fixed_ratios.size()) * (pow(param->densities[i], 2));
  }
  param->prior_num = num_priors;
  DDim shape_out({feature_height, feature_width, num_priors, 4});
  int32_t channel_size = feature_height * feature_width * num_priors * 4;
  boxes->Resize(shape_out);
  var->Resize(shape_out);
  int step_average = static_cast<int>((step_width + step_height) * 0.5);
  std::vector<float> sqrt_fixed_ratios;
  for (size_t i = 0; i < param->fixed_ratios.size(); i++) {
    sqrt_fixed_ratios.push_back(sqrt(param->fixed_ratios[i]));
  }
  float* cpu_data =
      static_cast<float*>(malloc(sizeof(float) * boxes->data_size() * 2));
  CHECK(cpu_data != nullptr);
  float* b_t = cpu_data;
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      float center_x = (w + param->offset) * step_width;
      float center_y = (h + param->offset) * step_height;

      for (size_t s = 0; s < param->fixed_sizes.size(); ++s) {
        auto fixed_size = param->fixed_sizes[s];
        int density = param->densities[s];
        int shift = step_average / density;
        // Generate density prior boxes with fixed ratios.
        for (size_t r = 0; r < param->fixed_ratios.size(); ++r) {
          float box_width_ratio = fixed_size * sqrt_fixed_ratios[r];
          float box_height_ratio = fixed_size / sqrt_fixed_ratios[r];
          float density_center_x = center_x - step_average / 2. + shift / 2.;
          float density_center_y = center_y - step_average / 2. + shift / 2.;
          for (int di = 0; di < density; ++di) {
            for (int dj = 0; dj < density; ++dj) {
              float center_x_temp = density_center_x + dj * shift;
              float center_y_temp = density_center_y + di * shift;
              b_t[0] = std::max(
                  (center_x_temp - box_width_ratio / 2.) / img_width, 0.);
              b_t[1] = std::max(
                  (center_y_temp - box_height_ratio / 2.) / img_height, 0.);
              b_t[2] = std::min(
                  (center_x_temp + box_width_ratio / 2.) / img_width, 1.);
              b_t[3] = std::min(
                  (center_y_temp + box_height_ratio / 2.) / img_height, 1.);
              b_t += 4;
            }
          }
        }
      }
    }
  }

  if (param->clip) {
    for (int32_t d = 0; d < channel_size; ++d) {
      cpu_data[d] = std::min(std::max(cpu_data[d], 0.f), 1.f);
    }
  }
  float* ptr = cpu_data + channel_size;
  int count = 0;
  for (int32_t h = 0; h < feature_height; ++h) {
    for (int32_t w = 0; w < feature_width; ++w) {
      for (int32_t i = 0; i < param->prior_num; ++i) {
        for (int j = 0; j < 4; ++j) {
          ptr[count] = param->variances[j];
          ++count;
        }
      }
    }
  }
  return cpu_data;
}

int DensityPriorBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  // inputs
  auto in_var_name = op_info->Input("Input").front();
  auto in = scope->FindVar(in_var_name)->GetMutable<lite::Tensor>();
  auto in_dims = in->dims();
  auto img_var_name = op_info->Input("Image").front();
  auto img = scope->FindVar(img_var_name)->GetMutable<lite::Tensor>();
  auto img_dims = img->dims();
  std::vector<int32_t> i_input_shape_data(in_dims.size());
  for (size_t i = 0; i < in_dims.size(); i++) {
    i_input_shape_data[i] = static_cast<int32_t>(in_dims[i]);
  }
  std::vector<int32_t> i_img_shape_data(img_dims.size());
  for (size_t i = 0; i < img_dims.size(); i++) {
    i_img_shape_data[i] = static_cast<int32_t>(img_dims[i]);
  }
  // outputs
  auto boxes_var_name = op_info->Output("Boxes").front();
  auto boxes = scope->FindVar(boxes_var_name)->GetMutable<lite::Tensor>();
  auto var_var_name = op_info->Output("Variances").front();
  // param
  st_priorbox_param param;
  param.clip = op_info->GetAttr<bool>("clip");
  param.flatten_to_2d = op_info->GetAttr<bool>("flatten_to_2d");
  param.fixed_sizes = op_info->GetAttr<std::vector<float>>("fixed_sizes");
  param.fixed_ratios = op_info->GetAttr<std::vector<float>>("fixed_ratios");
  param.variances = op_info->GetAttr<std::vector<float>>("variances");
  param.densities = op_info->GetAttr<std::vector<int>>("densities");
  param.offset = op_info->GetAttr<float>("offset");
  if (op_info->HasAttr("step_w")) {
    param.step_w = op_info->GetAttr<float>("step_w");
  }
  if (op_info->HasAttr("step_h")) {
    param.step_h = op_info->GetAttr<float>("step_h");
  }
  auto boxes_dims = boxes->dims();
  std::vector<int32_t> i_pri_out_shape_data(3);
  i_pri_out_shape_data[0] = 1;
  i_pri_out_shape_data[1] = 2;
  i_pri_out_shape_data[2] = boxes->data_size();
  auto bm_priorbox_name = lite::subgraph::bm::UniqueName("bm_priorbox");
  float* cpu_data = compute_density_priorbox_kernel(op, &param);
  boxes = scope->FindVar(boxes_var_name)->GetMutable<lite::Tensor>();
  i_pri_out_shape_data[2] = boxes->data_size();
#ifndef BM_DYNAMIC_COMPILE
  add_priorbox_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_input_shape_data[0]),
                     in_dims.size(),
                     static_cast<const char*>(in_var_name.c_str()),
                     const_cast<const int*>(&i_pri_out_shape_data[0]),
                     3,
                     static_cast<const char*>(bm_priorbox_name.c_str()),
                     static_cast<const float*>(cpu_data),
                     0,
                     nullptr,
                     0,
                     nullptr,
                     0,
                     nullptr,
                     0,
                     0,
                     0,
                     nullptr,
                     0,
                     0,
                     0.f,
                     0.f,
                     0.f);
#else
  free(cpu_data);
  user_cpu_param_t bm_param;
  bm_param.op_type = USER_PADDLE_DENSITY_PRIOR_BOX;
  CHECK_LE(param.fixed_sizes.size(), 20);
  bm_param.u.density_prior_box_param.fixed_sizes_len = param.fixed_sizes.size();
  memcpy(bm_param.u.density_prior_box_param.fixed_sizes,
         &param.fixed_sizes[0],
         param.fixed_sizes.size() * sizeof(float));

  CHECK_LE(param.fixed_ratios.size(), 20);
  bm_param.u.density_prior_box_param.fixed_ratios_len =
      param.fixed_ratios.size();
  memcpy(bm_param.u.density_prior_box_param.fixed_ratios,
         &param.fixed_ratios[0],
         param.fixed_ratios.size() * sizeof(float));

  CHECK_LE(param.densities.size(), 20);
  bm_param.u.density_prior_box_param.densities_len = param.densities.size();
  memcpy(bm_param.u.density_prior_box_param.densities,
         &param.densities[0],
         param.densities.size() * sizeof(int));

  CHECK_LE(param.variances.size(), 20);
  bm_param.u.density_prior_box_param.variances_len = param.variances.size();
  memcpy(bm_param.u.density_prior_box_param.variances,
         &param.variances[0],
         param.variances.size() * sizeof(float));
  bm_param.u.density_prior_box_param.step_w = param.step_w;
  bm_param.u.density_prior_box_param.step_h = param.step_h;
  bm_param.u.density_prior_box_param.offset = param.offset;
  bm_param.u.density_prior_box_param.prior_num = param.prior_num;
  bm_param.u.density_prior_box_param.clip = param.clip;
  bm_param.u.density_prior_box_param.flatten_to_2d = param.flatten_to_2d;

  int32_t* in_shape[2];
  int32_t in_dim[2];
  const char* in_name[2];
  in_shape[0] = &i_input_shape_data[0];
  in_shape[1] = &i_img_shape_data[0];
  in_dim[0] = in_dims.size();
  in_dim[1] = img_dims.size();
  in_name[0] = static_cast<const char*>(in_var_name.c_str());
  in_name[1] = static_cast<const char*>(img_var_name.c_str());
  int32_t* out_shape[1];
  int32_t out_dim[1];
  const char* out_name[1];
  out_shape[0] = &i_pri_out_shape_data[0];
  out_dim[0] = 3;
  out_name[0] = static_cast<const char*>(bm_priorbox_name.c_str());

  add_user_cpu_layer(graph->GetCompilerHandle(),
                     2,
                     in_shape,
                     in_dim,
                     in_name,
                     1,
                     out_shape,
                     out_dim,
                     out_name,
                     &bm_param,
                     static_cast<int>(sizeof(bm_param)));
#endif
  int32_t* shape[2];
  int32_t dim[2];
  const char* name[2];
  int32_t dim_size = 3;
  dim[0] = dim_size;
  dim[1] = dim_size;
  std::vector<int32_t> i_split_shape_data(dim_size);
  for (size_t i = 0; i < dim_size; i++) {
    i_split_shape_data[i] = i_pri_out_shape_data[i];
  }
  i_split_shape_data[1] /= 2;
  shape[0] = &i_split_shape_data[0];
  shape[1] = &i_split_shape_data[0];
  auto boxes_name = lite::subgraph::bm::UniqueName("bm_boxes");
  auto var_name = lite::subgraph::bm::UniqueName("bm_var");
  name[0] = static_cast<const char*>(boxes_name.c_str());
  name[1] = static_cast<const char*>(var_name.c_str());
  int split_size[2];
  split_size[0] = shape[0][1];
  split_size[1] = shape[1][1];
  add_tf_split_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_pri_out_shape_data[0]),
                     3,
                     static_cast<const char*>(bm_priorbox_name.c_str()),
                     2,
                     shape,
                     dim,
                     name,
                     3,
                     1,
                     split_size,
                     2);
  // final output
  std::vector<int32_t> i_output_shape_data(boxes_dims.size());
  for (size_t i = 0; i < boxes_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int32_t>(boxes_dims[i]);
  }
  add_reshape_layer_v2(graph->GetCompilerHandle(),
                       name[0],
                       shape[0],
                       3,
                       static_cast<const char*>(boxes_var_name.c_str()),
                       const_cast<const int*>(&i_output_shape_data[0]),
                       boxes_dims.size());
  add_reshape_layer_v2(graph->GetCompilerHandle(),
                       name[1],
                       shape[1],
                       3,
                       static_cast<const char*>(var_var_name.c_str()),
                       const_cast<const int*>(&i_output_shape_data[0]),
                       boxes_dims.size());
  graph->AddNode(boxes_var_name);
  graph->AddNode(var_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(density_prior_box,
                         kBM,
                         paddle::lite::subgraph::bm::DensityPriorBoxConverter);
