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

#include "lite/core/mir/elimination/ssd_boxes_calc_offline_pass.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

void SSDBoxesCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemovePriorboxPattern(graph);
  RemoveReshapePattern(graph);
  RemoveFlattenPattern(graph);
  RemoveConcatPattern(graph);
}

void SSDBoxesCalcOfflinePass::RemovePriorboxPattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().picked_kernel().op_type() != "prior_box") continue;

    std::set<const Node*> nodes2rm_;
    auto& priorbox_instruct = node->AsStmt();
    auto* scope = priorbox_instruct.op()->scope();
    auto op_desc = priorbox_instruct.mutable_op_info();

    // Get priorbox's input tensor
    auto image_var = scope->FindVar(op_desc->Input("Image").front());
    auto image_t = &(image_var->Get<lite::Tensor>());
    auto input_var = scope->FindVar(op_desc->Input("Input").front());
    auto input_t = &(input_var->Get<lite::Tensor>());
    auto img_h = image_t->dims()[2];
    auto img_w = image_t->dims()[3];
    // Get priorbox's output tensor
    auto boxes_var = scope->FindVar(op_desc->Output("Boxes").front());
    auto boxes_t = boxes_var->GetMutable<lite::Tensor>();
    auto variances_var = scope->FindVar(op_desc->Output("Variances").front());
    auto variances_t = variances_var->GetMutable<lite::Tensor>();
    // Get priorbox's other attr
    auto is_flip = true;
    if (op_desc->HasAttr("flip")) {
      is_flip = op_desc->GetAttr<bool>("flip");
    }
    auto is_clip = true;
    if (op_desc->HasAttr("clip")) {
      is_clip = op_desc->GetAttr<bool>("clip");
    }
    auto min_max_aspect_ratios_order = false;
    if (op_desc->HasAttr("min_max_aspect_ratios_order")) {
      min_max_aspect_ratios_order =
          op_desc->GetAttr<bool>("min_max_aspect_ratios_order");
    }
    auto max_sizes = op_desc->GetAttr<std::vector<float>>("max_sizes");
    auto min_sizes = op_desc->GetAttr<std::vector<float>>("min_sizes");
    auto aspect_ratios = op_desc->GetAttr<std::vector<float>>("aspect_ratios");
    std::vector<float> aspect_ratios_vec;
    ExpandAspectRatios(aspect_ratios, is_flip, &aspect_ratios_vec);
    auto variances = op_desc->GetAttr<std::vector<float>>("variances");
    auto step_h = 0.f;
    if (op_desc->HasAttr("step_h")) {
      step_h = op_desc->GetAttr<float>("step_h");
    }
    auto step_w = 0.f;
    if (op_desc->HasAttr("step_w")) {
      step_w = op_desc->GetAttr<float>("step_w");
    }
    auto offset = 0.5f;
    if (op_desc->HasAttr("offset")) {
      offset = op_desc->GetAttr<float>("offset");
    }
    int prior_num =
        (aspect_ratios_vec.size() * min_sizes.size()) + max_sizes.size();
    const std::vector<std::string> order_tmp;
    // Calc priorbox
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
    // Offline calc priorbox, only retain output tensor as persistable tensor
    boxes_t->set_persistable(true);
    variances_t->set_persistable(true);
    auto priorbox_outlinks = node->outlinks;
    for (auto& priorbox_out_link : priorbox_outlinks) {
      priorbox_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

void SSDBoxesCalcOfflinePass::RemoveFlattenPattern(
    const std::unique_ptr<SSAGraph>& graph) {
  auto check_flatten_after_priorbox = [](Node* p) -> bool {
    auto check_flatten_inlinks = p->inlinks;
    for (auto& check_flatten_in_link : check_flatten_inlinks) {
      if (check_flatten_in_link->arg()->is_weight != true) {
        return false;
      }
    }
    return true;
  };

  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().picked_kernel().op_type() != "flatten" &&
        node->AsStmt().picked_kernel().op_type() != "flatten2")
      continue;
    if (check_flatten_after_priorbox(node) != true) continue;

    std::set<const Node*> nodes2rm_;
    auto& flatten_instruct = node->AsStmt();
    auto* scope = flatten_instruct.op()->scope();
    auto op_desc = flatten_instruct.mutable_op_info();

    auto input_var = scope->FindVar(op_desc->Input("X").front());
    auto input_t = &(input_var->Get<lite::Tensor>());
    auto output_var = scope->FindVar(op_desc->Output("Out").front());
    auto output_t = output_var->GetMutable<lite::Tensor>();
    // Calc flatten offline
    ComputeFlatten(input_t, output_t);
    // Offline calc reshape, only retain output tensor as persistable tensor
    output_t->set_persistable(true);

    auto flatten_outlinks = node->outlinks;
    for (auto& flatten_out_link : flatten_outlinks) {
      flatten_out_link->arg()->is_weight = true;
    }
    auto flatten_inlinks = node->inlinks;
    for (auto& flatten_in_link : flatten_inlinks) {
      nodes2rm_.insert(flatten_in_link);
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

void SSDBoxesCalcOfflinePass::RemoveReshapePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  auto check_reshape_after_priorbox = [](Node* p) -> bool {
    auto check_reshape_inlinks = p->inlinks;
    for (auto& check_reshape_in_link : check_reshape_inlinks) {
      if (check_reshape_in_link->arg()->is_weight != true) {
        return false;
      }
    }
    return true;
  };

  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().picked_kernel().op_type() != "reshape" &&
        node->AsStmt().picked_kernel().op_type() != "reshape2")
      continue;
    if (check_reshape_after_priorbox(node) != true) continue;

    std::set<const Node*> nodes2rm_;
    auto& reshape_instruct = node->AsStmt();
    auto* scope = reshape_instruct.op()->scope();
    auto op_desc = reshape_instruct.mutable_op_info();

    auto input_var = scope->FindVar(op_desc->Input("X").front());
    auto input_t = &(input_var->Get<lite::Tensor>());
    auto output_var = scope->FindVar(op_desc->Output("Out").front());
    auto output_t = output_var->GetMutable<lite::Tensor>();
    // Calc reshape offline
    ComputeReshape(input_t, output_t);
    // Offline calc reshape, only retain output tensor as persistable tensor
    output_t->set_persistable(true);

    auto reshape_outlinks = node->outlinks;
    for (auto& reshape_out_link : reshape_outlinks) {
      reshape_out_link->arg()->is_weight = true;
    }
    auto reshape_inlinks = node->inlinks;
    for (auto& reshape_in_link : reshape_inlinks) {
      nodes2rm_.insert(reshape_in_link);
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

void SSDBoxesCalcOfflinePass::RemoveConcatPattern(
    const std::unique_ptr<SSAGraph>& graph) {
  auto check_concat_after_reshape = [](Node* p) -> bool {
    auto check_concat_inlinks = p->inlinks;
    for (auto& check_concat_in_link : check_concat_inlinks) {
      if (check_concat_in_link->arg()->is_weight != true) {
        return false;
      }
    }
    return true;
  };

  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().picked_kernel().op_type() != "concat") continue;
    if (check_concat_after_reshape(node) != true) continue;

    std::set<const Node*> nodes2rm_;
    auto& concat_instruct = node->AsStmt();
    auto* scope = concat_instruct.op()->scope();
    auto op_desc = concat_instruct.mutable_op_info();

    std::vector<lite::Tensor*> inputs_tensors;

    for (auto* in_var_node : node->inlinks) {
      auto concat_in_tensor =
          scope->FindVar(in_var_node->AsArg().name)->GetMutable<lite::Tensor>();
      inputs_tensors.push_back(concat_in_tensor);
    }

    // Get concat's output tensor
    auto output_var = scope->FindVar(op_desc->Output("Out").front());
    auto output_t = output_var->GetMutable<lite::Tensor>();

    // get concat's other attr
    auto axis = op_desc->GetAttr<int>("axis");
    if (axis != 0) {
      LOG(WARNING) << "the ssd priorbox concat's axis must be 0 ";
    }

    // Calc the concat offline
    ComputeConcat(inputs_tensors, output_t);
    // Offline calc reshape, only retain output tensor as persistable tensor
    output_t->set_persistable(true);

    auto concat_outlinks = node->outlinks;
    for (auto& concat_out_link : concat_outlinks) {
      concat_out_link->arg()->is_weight = true;
    }
    auto concat_inlinks = node->inlinks;
    for (auto& concat_in_link : concat_inlinks) {
      nodes2rm_.insert(concat_in_link);
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

void SSDBoxesCalcOfflinePass::ExpandAspectRatios(
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

void SSDBoxesCalcOfflinePass::ComputePriorbox(
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
  // Compute output shape
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
      float* min_buf =
          reinterpret_cast<float*>(host::malloc(sizeof(float) * 4));
      float* max_buf =
          reinterpret_cast<float*>(host::malloc(sizeof(float) * 4));
      float* com_buf = reinterpret_cast<float*>(
          host::malloc(sizeof(float) * aspect_ratio_.size() * 4));

      for (auto s = 0; s < min_size_.size(); ++s) {
        int min_idx = 0;
        int max_idx = 0;
        int com_idx = 0;
        int min_size = min_size_[s];
        // First prior: aspect_ratio = 1, size = min_size
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
          //! Second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
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

        //! Rest of priors
        for (auto r = 0; r < aspect_ratio_.size(); ++r) {
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
      host::free(min_buf);
      host::free(max_buf);
      host::free(com_buf);
    }
  }
  //! Clip the prior's coordinate such that it is within [0, 1]
  if (is_clip_) {
    for (int d = 0; d < channel_size; ++d) {
      _cpu_data[d] = std::min(std::max(_cpu_data[d], 0.f), 1.f);
    }
  }
  //! Set the variance.
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

void SSDBoxesCalcOfflinePass::ComputeFlatten(const lite::Tensor* in,
                                             lite::Tensor* out) {
  // In CopyDataFrom, the target tensor's dims will be set to the source
  // tensor's dims.
  auto out_dims = out->dims();
  auto out_lod = out->lod();
  out->CopyDataFrom(*in);
  out->Resize(out_dims);
  out->set_lod(out_lod);
}

void SSDBoxesCalcOfflinePass::ComputeReshape(const lite::Tensor* in,
                                             lite::Tensor* out) {
  // In CopyDataFrom, the target tensor's dims will be set to the source
  // tensor's dims.
  auto out_dims = out->dims();
  out->CopyDataFrom(*in);
  out->Resize(out_dims);
}

std::vector<size_t> SSDBoxesCalcOfflinePass::StrideNumel(const DDim& ddim) {
  std::vector<size_t> strides(ddim.size());
  strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return strides;
}

void SSDBoxesCalcOfflinePass::ComputeConcat(
    const std::vector<lite::Tensor*> inputs, lite::Tensor* output) {
  size_t output_offset = 0;
  for (auto* in : inputs) {
    auto in_stride = StrideNumel(in->dims());
    auto out_stride = StrideNumel(output->dims());
    void* dst = output->mutable_data<float>() + output_offset;
    const void* src = in->data<float>();
    // Src and dst tensor should have the same dims size.
    CHECK(in_stride.size() == out_stride.size());
    std::memcpy(dst, src, sizeof(float) * in_stride[0]);
    output_offset += in_stride[0];
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(ssd_boxes_calc_offline_pass,
                  paddle::lite::mir::SSDBoxesCalcOfflinePass)
    .BindTargets(
        {TARGET(kRKNPU), TARGET(kNPU), TARGET(kOpenCL), TARGET(kNNAdapter)});
