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

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

class PriorboxEliminator : public FuseBase {
 public:
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 protected:
  void DeleteInterNodes(SSAGraph* graph) override;

 private:
  void* fast_malloc(size_t size);
  void fast_free(void* ptr);
  void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                          bool flip,
                          std::vector<float>* output_aspect_ratior);
  void ComputePriorbox(const lite::Tensor* input,
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
                       bool min_max_aspect_ratios_order);
  std::set<const Node*> nodes2rm_;
};

class PriorboxEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
