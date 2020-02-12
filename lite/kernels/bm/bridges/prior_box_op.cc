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
  param->prior_num = param->aspect_ratios.size() * param->min_sizes.size();
  if (param->max_sizes.size() > 0) {
    param->prior_num += param->max_sizes.size();
  }
  int32_t win1 = in_dims[3];
  int32_t hin1 = in_dims[2];
  DDim shape_out({hin1, win1, param->prior_num, 4});
  boxes->Resize(shape_out);
  var->Resize(shape_out);
  // boxes->mutable_data<float>();
  // var->mutable_data<float>();
  float* cpu_data =
      static_cast<float*>(malloc(sizeof(float) * boxes->data_size() * 2));
  CHECK(cpu_data != nullptr);
  const int32_t width = in_dims[3];
  const int32_t height = in_dims[2];
  int32_t img_width = param->img_w;
  int32_t img_height = param->img_h;
  if (img_width == 0 || img_height == 0) {
    img_width = img_dims[3];
    img_height = img_dims[2];
  }
  float step_w = param->step_w;
  float step_h = param->step_h;
  if (step_w == 0.f || step_h == 0.f) {
    step_w = static_cast<float>(img_width) / width;
    step_h = static_cast<float>(img_height) / height;
  }
  float offset = param->offset;
  int32_t channel_size = height * width * param->prior_num * 4;
  int32_t idx = 0;
  ///////////////////////////////////////////////////////////////////////
  for (int32_t h = 0; h < height; ++h) {
    for (int32_t w = 0; w < width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width = 0.f;
      float box_height = 0.f;
      float* min_buf = reinterpret_cast<float*>(malloc(sizeof(float) * 4));
      float* max_buf = reinterpret_cast<float*>(malloc(sizeof(float) * 4));
      float* com_buf = reinterpret_cast<float*>(
          malloc(sizeof(float) * expand_aspect_ratios.size() * 4));
      CHECK(min_buf != nullptr);
      CHECK(max_buf != nullptr);
      CHECK(com_buf != nullptr);
      // LOG(INFO) << "the number of min_size is " << min_sizes_.size();
      for (size_t s = 0; s < param->min_sizes.size(); ++s) {
        int32_t min_idx = 0;
        int32_t max_idx = 0;
        int32_t com_idx = 0;
        int32_t min_size = param->min_sizes[s];
        //! first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size;
        //! xmin
        min_buf[min_idx++] = (center_x - box_width / 2.f) / img_width;
        //! ymin
        min_buf[min_idx++] = (center_y - box_height / 2.f) / img_height;
        //! xmax
        min_buf[min_idx++] = (center_x + box_width / 2.f) / img_width;
        //! ymax
        min_buf[min_idx++] = (center_y + box_height / 2.f) / img_height;
        if (param->max_sizes.size() > 0) {
          int max_size = param->max_sizes[s];
          //! second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrtf(min_size * max_size);
          //! xmin
          max_buf[max_idx++] = (center_x - box_width / 2.f) / img_width;
          //! ymin
          max_buf[max_idx++] = (center_y - box_height / 2.f) / img_height;
          //! xmax
          max_buf[max_idx++] = (center_x + box_width / 2.f) / img_width;
          //! ymax
          max_buf[max_idx++] = (center_y + box_height / 2.f) / img_height;
        }
        //! rest of priors
        for (size_t r = 0; r < expand_aspect_ratios.size(); ++r) {
          float ar = expand_aspect_ratios[r];
          if (fabs(ar - 1.) < 1e-6) {
            continue;
          }
          box_width = min_size * sqrt(ar);
          box_height = min_size / sqrt(ar);
          //! xmin
          com_buf[com_idx++] = (center_x - box_width / 2.f) / img_width;
          //! ymin
          com_buf[com_idx++] = (center_y - box_height / 2.f) / img_height;
          //! xmax
          com_buf[com_idx++] = (center_x + box_width / 2.f) / img_width;
          //! ymax
          com_buf[com_idx++] = (center_y + box_height / 2.f) / img_height;
        }
        if (param->min_max_aspect_ratios_order) {
          memcpy(cpu_data + idx, min_buf, sizeof(float) * min_idx);
          idx += min_idx;
          memcpy(cpu_data + idx, max_buf, sizeof(float) * max_idx);
          idx += max_idx;
          memcpy(cpu_data + idx, com_buf, sizeof(float) * com_idx);
          idx += com_idx;
        } else {
          memcpy(cpu_data + idx, com_buf, sizeof(float) * com_idx);
          idx += com_idx;
          memcpy(cpu_data + idx, max_buf, sizeof(float) * max_idx);
          idx += max_idx;
        }
      }
      free(min_buf);
      free(max_buf);
      free(com_buf);
    }
  }
  //! clip the prior's coordidate such that it is within [0, 1]
  if (param->clip) {
    for (int32_t d = 0; d < channel_size; ++d) {
      cpu_data[d] = std::min(std::max(cpu_data[d], 0.f), 1.f);
    }
  }
  //! set the variance.
  float* ptr = cpu_data + channel_size;
  int count = 0;
  for (int32_t h = 0; h < height; ++h) {
    for (int32_t w = 0; w < width; ++w) {
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
  auto unique_op_name = lite::subgraph::bm::UniqueName(op_type);
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
  if (op_info->HasAttr("min_max_aspect_ratios_order")) {
    param.min_max_aspect_ratios_order =
        op_info->GetAttr<bool>("min_max_aspect_ratios_order");
  }
  float* cpu_data = compute_priorbox_kernel(op, &param);
  compute_priorbox_kernel(op, param);
  auto boxes_dims = boxes->dims();
  std::vector<int32_t> i_pri_out_shape_data(boxes_dims.size());
  for (size_t i = 0; i < boxes_dims.size(); i++) {
    i_pri_out_shape_data[i] = static_cast<int32_t>(boxes_dims[i]);
  }
  i_pri_out_shape_data[0] *= 2;
  add_priorbox_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_input_shape_data[0]),
                     in_dims.size(),
                     static_cast<const char*>(in_var_name.c_str()),
                     const_cast<const int*>(&i_pri_out_shape_data[0]),
                     boxes_dims.size(),
                     static_cast<const char*>(unique_op_name.c_str()),
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
  std::vector<int32_t> i_output_shape_data(boxes_dims.size());
  for (size_t i = 0; i < boxes_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int32_t>(boxes_dims[i]);
  }
  int32_t* shape[2];
  int dim[2];
  const char* name[2];
  dim[0] = boxes_dims.size();
  dim[1] = boxes_dims.size();
  name[0] = static_cast<const char*>(boxes_var_name.c_str());
  name[1] = static_cast<const char*>(var_var_name.c_str());
  shape[0] = &i_output_shape_data[0];
  shape[1] = &i_output_shape_data[0];
  int split_size = 2;
  add_tf_split_layer(graph->GetCompilerHandle(),
                     const_cast<const int*>(&i_pri_out_shape_data[0]),
                     boxes_dims.size(),
                     static_cast<const char*>(unique_op_name.c_str()),
                     2,
                     shape,
                     dim,
                     name,
                     boxes_dims.size(),
                     0,
                     &split_size,
                     0);
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
