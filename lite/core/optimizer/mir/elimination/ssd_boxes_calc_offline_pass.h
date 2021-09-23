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

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

// Prior_box/Density_prior_box don't depend on feature-map data, only depend on
// image & feature-map size,
// so if the shape is determined, we can calculate it offline in opt stage,
// and the reshape(2) & flatten(2) & concat which linked with prior-box can
// be calculate offline too.
//
// For example:
//   image-size            feature-size       image-size            feature-size
//       |                       |                 |                       |
//       |                       |                 |                       |
//       |                       |                 |                       |
//       |                       |                 |                       |
//       ----- OP: prior_box ----                  ------OP: prior_box-----
//          or depsity_prior_box                      or depsity_prior_box
//                  |                                           |
//                  |                                           |
//                  |                                           |
//                  |                                           |
//     boxes----------------variances            boxes----------------variances
//       |                      |                  |                      |
//       |                      |                  |                      |
//       |                      |                  |                      |
//       |                      |                  |                      |
// OP:reshape|flatten  OP:reshape|flatten  OP:reshape|flatten OP:reshape|flatten
//       |                      |                  |                      |
//       |                      |                  |                      |
//       |                      |                  |                      |
//       |                      |                  |                      |
//  output                   output              output                output
//       |                      |                  |                      |
//       |                      |                  |                      |
//       |                      |                  |                      |
//       |                      |                  |                      |
//       ------ OP: concat ------------------------                       |
//                   |          |                                         |
//                   |          |                                         |
//                   |          |                                         |
//                   |          |                                         |
//                   |          |------------------ OP: concat -----------|
//                   |                                    |
//                   |                                    |
//                   |                                    |
//                   |                                    |
//                 boxes                             variances
//                   |                                    |
//                   |                                    |
//                   |                                    |
//                   |                                    |
//                   ----------- OP: box coder -----------
//                                     |
//                                     |
//                                     v
//
// After the pass is applied:
//                 boxes                             variances
//                   |                                    |
//                   |                                    |
//                   |                                    |
//                   |                                    |
//                   ----------- OP: box coder -----------
//                                     |
//                                     |
//                                     v

class SSDBoxesCalcOfflinePass : public mir::StmtPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  void RemovePriorboxPattern(const std::unique_ptr<SSAGraph>& graph);
  void RemoveFlattenPattern(const std::unique_ptr<SSAGraph>& graph);
  void RemoveReshapePattern(const std::unique_ptr<SSAGraph>& graph);
  void RemoveConcatPattern(const std::unique_ptr<SSAGraph>& graph);

 private:
  void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                          bool flip,
                          std::vector<float>* output_aspect_ratior);
  void ComputeDensityPriorBox(const lite::Tensor* input,
                              const lite::Tensor* image,
                              lite::Tensor** boxes,
                              lite::Tensor** variances,
                              const std::vector<float>& min_size_,
                              const std::vector<float>& fixed_size_,
                              const std::vector<float>& fixed_ratio_,
                              const std::vector<int>& density_size_,
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
  void ComputeReshape(const lite::Tensor* in, lite::Tensor* out);
  void ComputeFlatten(const lite::Tensor* in, lite::Tensor* out);
  void ComputeConcat(const std::vector<lite::Tensor*> inputs,
                     lite::Tensor* output);
  std::vector<size_t> StrideNumel(const DDim& ddim);
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
