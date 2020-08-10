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
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

typedef struct __tag_st_priorbox_param {
  std::vector<float> min_sizes;
  std::vector<float> max_sizes;
  std::vector<float> aspect_ratios;
  std::vector<float> variances;
  float step_w;
  float step_h;
  float offset;
  int32_t img_w;
  int32_t img_h;
  int32_t prior_num;
  bool min_max_aspect_ratios_order;
  bool clip;
  bool flip;
} st_priorbox_param;

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

float* compute_priorbox_kernel(OpLite* op, st_priorbox_param* param) {
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
  std::vector<float> expand_aspect_ratios;
  ExpandAspectRatios(param->aspect_ratios, param->flip, &expand_aspect_ratios);
  param->aspect_ratios.clear();
  for (size_t i = 0; i < expand_aspect_ratios.size(); i++) {
    param->aspect_ratios.push_back(expand_aspect_ratios[i]);
  }

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
  int num_priors = param->aspect_ratios.size() * param->min_sizes.size();
  if (param->max_sizes.size() > 0) {
    num_priors += param->max_sizes.size();
  }
  param->prior_num = num_priors;
  DDim shape_out({feature_height, feature_width, num_priors, 4});
  int32_t channel_size = feature_height * feature_width * num_priors * 4;
  boxes->Resize(shape_out);
  var->Resize(shape_out);
  float* cpu_data =
      static_cast<float*>(malloc(sizeof(float) * boxes->data_size() * 2));
  CHECK(cpu_data != nullptr);
  float* b_t = cpu_data;
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      float center_x = (w + param->offset) * step_width;
      float center_y = (h + param->offset) * step_height;
      float box_width, box_height;
      for (size_t s = 0; s < param->min_sizes.size(); ++s) {
        auto min_size = param->min_sizes[s];
        if (param->min_max_aspect_ratios_order) {
          box_width = box_height = min_size / 2.;
          b_t[0] = (center_x - box_width) / img_width;
          b_t[1] = (center_y - box_height) / img_height;
          b_t[2] = (center_x + box_width) / img_width;
          b_t[3] = (center_y + box_height) / img_height;
          b_t += 4;
          if (param->max_sizes.size() > 0) {
            auto max_size = param->max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
          // priors with different aspect ratios
          for (size_t r = 0; r < param->aspect_ratios.size(); ++r) {
            float ar = param->aspect_ratios[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
        } else {
          // priors with different aspect ratios
          for (size_t r = 0; r < param->aspect_ratios.size(); ++r) {
            float ar = param->aspect_ratios[r];
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
          if (param->max_sizes.size() > 0) {
            auto max_size = param->max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
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

int PriorBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  // outputs
  auto boxes_var_name = op_info->Output("Boxes").front();
  auto boxes = scope->FindVar(boxes_var_name)->GetMutable<lite::Tensor>();
  auto var_var_name = op_info->Output("Variances").front();
  // param
  st_priorbox_param param;
  param.clip = op_info->GetAttr<bool>("clip");
  param.min_sizes = op_info->GetAttr<std::vector<float>>("min_sizes");
  param.max_sizes = op_info->GetAttr<std::vector<float>>("max_sizes");
  param.aspect_ratios = op_info->GetAttr<std::vector<float>>("aspect_ratios");
  param.variances = op_info->GetAttr<std::vector<float>>("variances");
  param.offset = op_info->GetAttr<float>("offset");
  if (op_info->HasAttr("flip")) {
    param.flip = op_info->GetAttr<bool>("flip");
  }
  if (op_info->HasAttr("img_w")) {
    param.img_w = op_info->GetAttr<int32_t>("img_w");
  }
  if (op_info->HasAttr("img_h")) {
    param.img_h = op_info->GetAttr<int32_t>("img_h");
  }
  if (op_info->HasAttr("step_w")) {
    param.step_w = op_info->GetAttr<float>("step_w");
  }
  if (op_info->HasAttr("step_h")) {
    param.step_h = op_info->GetAttr<float>("step_h");
  }
  if (op_info->HasAttr("prior_num")) {
    param.prior_num = op_info->GetAttr<int32_t>("prior_num");
  }
  param.min_max_aspect_ratios_order = false;
  if (op_info->HasAttr("min_max_aspect_ratios_order")) {
    param.min_max_aspect_ratios_order =
        op_info->GetAttr<bool>("min_max_aspect_ratios_order");
  }
  float* cpu_data = compute_priorbox_kernel(op, &param);
  auto boxes_dims = boxes->dims();
  std::vector<int32_t> i_pri_out_shape_data(3);
  i_pri_out_shape_data[0] = 1;
  i_pri_out_shape_data[1] = 2;
  i_pri_out_shape_data[2] = boxes->data_size();
  auto bm_priorbox_name = lite::subgraph::bm::UniqueName("bm_priorbox");
  add_priorbox_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_input_shape_data[0]),
                     in_dims.size(),
                     static_cast<const char*>(in_var_name.c_str()),
                     const_cast<const int*>(&i_pri_out_shape_data[0]),
                     3,
                     static_cast<const char*>(bm_priorbox_name.c_str()),
                     static_cast<const float*>(cpu_data),
                     param.min_sizes.size(),
                     const_cast<const float*>(&param.min_sizes[0]),
                     param.max_sizes.size(),
                     const_cast<const float*>(&param.max_sizes[0]),
                     param.aspect_ratios.size(),
                     const_cast<const float*>(&param.aspect_ratios[0]),
                     static_cast<int>(param.flip),
                     static_cast<int>(param.clip),
                     param.variances.size(),
                     const_cast<const float*>(&param.variances[0]),
                     param.img_h,
                     param.img_w,
                     param.step_h,
                     param.step_w,
                     param.offset);
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

REGISTER_SUBGRAPH_BRIDGE(prior_box,
                         kBM,
                         paddle::lite::subgraph::bm::PriorBoxConverter);
