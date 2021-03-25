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

#include "lite/core/mir/elimination/priorbox_eliminate_pass.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
const int MALLOC_ALIGN = 64;

void* PriorboxEliminator::fast_malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(malloc(offset + size));
  if (!p) {
    return nullptr;
  }
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  memset(r, 0, size);
  return r;
}

void PriorboxEliminator::fast_free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

void PriorboxEliminator::ExpandAspectRatios(
    const std::vector<float>& input_aspect_ratior,
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

void PriorboxEliminator::ComputePriorbox(
    const lite::Tensor* input,
    const lite::Tensor* image,
    lite::Tensor** boxes,
    lite::Tensor** variances,
    const std::vector<float>& min_size_,
    const std::vector<float>& max_size_,
    const std::vector<float>& aspect_ratio_,
    const std::vector<float>& variance_,
    int img_w_,
    int img_h_,
    float step_w_,
    float step_h_,
    float offset_,
    int prior_num_,
    bool is_flip_,
    bool is_clip_,
    const std::vector<std::string>& order_,
    bool min_max_aspect_ratios_order) {
  // compute output shape
  int win1 = input->dims()[3];
  int hin1 = input->dims()[2];
  DDim shape_out({hin1, win1, prior_num_, 4});
  (*boxes)->Resize(shape_out);
  (*variances)->Resize(shape_out);

  float* _cpu_data = (*boxes)->mutable_data<float>();
  float* _variance_data = (*variances)->mutable_data<float>();

  const int width = win1;
  const int height = hin1;
  int img_width = img_w_;
  int img_height = img_h_;
  if (img_width == 0 || img_height == 0) {
    img_width = image->dims()[3];
    img_height = image->dims()[2];
  }
  float step_w = step_w_;
  float step_h = step_h_;
  if (step_w == 0 || step_h == 0) {
    step_w = static_cast<float>(img_width) / width;
    step_h = static_cast<float>(img_height) / height;
  }
  float offset = offset_;
  int channel_size = height * width * prior_num_ * 4;
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width;
      float box_height;
      float* min_buf = reinterpret_cast<float*>(fast_malloc(sizeof(float) * 4));
      float* max_buf = reinterpret_cast<float*>(fast_malloc(sizeof(float) * 4));
      float* com_buf = reinterpret_cast<float*>(
          fast_malloc(sizeof(float) * aspect_ratio_.size() * 4));

      for (int s = 0; s < min_size_.size(); ++s) {
        int min_idx = 0;
        int max_idx = 0;
        int com_idx = 0;
        int min_size = min_size_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size;
        //! xmin
        min_buf[min_idx++] = (center_x - box_width / 2.f) / img_width;
        //! ymin
        min_buf[min_idx++] = (center_y - box_height / 2.f) / img_height;
        //! xmax
        min_buf[min_idx++] = (center_x + box_width / 2.f) / img_width;
        //! ymax
        min_buf[min_idx++] = (center_y + box_height / 2.f) / img_height;

        if (max_size_.size() > 0) {
          int max_size = max_size_[s];
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
        for (int r = 0; r < aspect_ratio_.size(); ++r) {
          float ar = aspect_ratio_[r];
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
        if (min_max_aspect_ratios_order) {
          memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
          idx += min_idx;
          memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
          idx += max_idx;
          memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
          idx += com_idx;
        } else {
          memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
          idx += min_idx;
          memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
          idx += com_idx;
          memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
          idx += max_idx;
        }
      }
      fast_free(min_buf);
      fast_free(max_buf);
      fast_free(com_buf);
    }
  }
  //! clip the prior's coordinate such that it is within [0, 1]
  if (is_clip_) {
    for (int d = 0; d < channel_size; ++d) {
      _cpu_data[d] = std::min(std::max(_cpu_data[d], 0.f), 1.f);
    }
  }
  //! set the variance.
  int count = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int i = 0; i < prior_num_; ++i) {
        for (int j = 0; j < 4; ++j) {
          _variance_data[count] = variance_[j];
          ++count;
        }
      }
    }
  }
}

void PriorboxEliminator::BuildPattern() {
  // prior_box #0 node
  auto* prior_box = OpNode("prior_box", "prior_box");
  // prior_box #0 input
  auto* prior_box_input_x =
      VarNode("prior_box_input_x")->assert_is_op_input("prior_box", "Input");
  auto* prior_box_input_image = VarNode("prior_box_input_image")
                                    ->assert_is_op_input("prior_box", "Image");
  // prior_box #0 output
  auto* prior_box_output_boxes =
      VarNode("prior_box_output_boxes")
          ->assert_is_op_output("prior_box", "Boxes");
  auto* prior_box_output_var =
      VarNode("prior_box_output_var")
          ->assert_is_op_output("prior_box", "Variances");

  // prior_box #0 topology
  std::vector<PMNode*> prior_box_inputs{prior_box_input_x,
                                        prior_box_input_image};
  std::vector<PMNode*> prior_box_outputs{prior_box_output_boxes,
                                         prior_box_output_var};
  prior_box_inputs >> *prior_box >> prior_box_outputs;
}

void PriorboxEliminator::DeleteInterNodes(SSAGraph* graph) {
  GraphSafeRemoveNodes(graph, nodes2rm_);
}

void PriorboxEliminator::InsertNewNode(SSAGraph* graph,
                                       const key2nodes_t& matched) {
  auto priorbox_instruct = matched.at("prior_box")->stmt();
  auto op_desc = priorbox_instruct->mutable_op_info();
  auto* scope = priorbox_instruct->op()->scope();
  // get priorbox's input tensor
  auto image_var = scope->FindVar(op_desc->Input("Image").front());
  auto image_t = &(image_var->Get<lite::Tensor>());
  auto input_var = scope->FindVar(op_desc->Input("Input").front());
  auto input_t = &(input_var->Get<lite::Tensor>());
  auto img_h = image_t->dims()[2];
  auto img_w = image_t->dims()[3];
  // get priorbox's output tensor
  auto boxes_var = scope->FindVar(op_desc->Output("Boxes").front());
  auto boxes_t = boxes_var->GetMutable<lite::Tensor>();
  auto variances_var = scope->FindVar(op_desc->Output("Variances").front());
  auto variances_t = variances_var->GetMutable<lite::Tensor>();
  // get priorbox's other attr
  auto is_clip = op_desc->GetAttr<bool>("clip");
  auto is_flip = op_desc->GetAttr<bool>("flip");
  auto min_max_aspect_ratios_order =
      op_desc->GetAttr<bool>("min_max_aspect_ratios_order");
  auto max_sizes = op_desc->GetAttr<std::vector<float>>("max_sizes");
  auto min_sizes = op_desc->GetAttr<std::vector<float>>("min_sizes");
  auto aspect_ratios = op_desc->GetAttr<std::vector<float>>("aspect_ratios");
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratios, is_flip, &aspect_ratios_vec);
  auto variances = op_desc->GetAttr<std::vector<float>>("variances");
  auto step_h = op_desc->GetAttr<float>("step_h");
  auto step_w = op_desc->GetAttr<float>("step_w");
  auto offset = op_desc->GetAttr<float>("offset");
  int prior_num =
      (aspect_ratios_vec.size() * min_sizes.size()) + max_sizes.size();
  const std::vector<std::string> order_tmp;
  // calcu priorbox
  ComputePriorbox(input_t,
                  image_t,
                  &boxes_t,
                  &variances_t,
                  min_sizes,
                  max_sizes,
                  aspect_ratios_vec,
                  variances,
                  img_w,
                  img_h,
                  step_w,
                  step_h,
                  offset,
                  prior_num,
                  is_flip,
                  is_clip,
                  order_tmp,
                  min_max_aspect_ratios_order);
  // set the output as persistable-tensor
  boxes_t->set_persistable(true);
  variances_t->set_persistable(true);
  auto output_boxes_node = matched.at("prior_box_output_boxes");
  output_boxes_node->arg()->is_weight = true;
  auto output_var_node = matched.at("prior_box_output_var");
  output_var_node->arg()->is_weight = true;
  nodes2rm_.insert(matched.at("prior_box"));
}

void PriorboxEliminatePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  PriorboxEliminator eliminator;
  eliminator(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_priorbox_eliminate_pass,
                  paddle::lite::mir::PriorboxEliminatePass)
    .BindTargets({TARGET(kNPU), TARGET(kRKNPU)});
