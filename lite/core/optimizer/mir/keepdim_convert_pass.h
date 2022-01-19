// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * KeepdimConvertPass splits some ops whose attribute `keepdims` or `keep_dim`
 * == false to two ops.
 * The reason for adding this pass is that it is hard for gpu to do reshape
 * opterations in arg_max/reduce_mean, etc,. So we split this problem.
 *
 * For example:
 *        |
 *        v
 *     arg_max(keepdim=false)
 *        |
 *        v
 *
 * After this pass is applied:
 *        |
 *        v
 *     arg_max(keepdim=true)
 *        |
 *        v
 *     reshape
 *        |
 *        v
 */

class KeepdimConvertPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  void ComplementInputs(SSAGraph* graph,
                        Node* inst_node,
                        Node* in,
                        std::map<std::string, Node*>* copied_nodes);

  void AddReshapeInst(const Type& from,
                      const Type& to,
                      Node* in,
                      SSAGraph* graph,
                      Node* inst_node,
                      std::map<std::string, Node*>* copied_nodes,
                      const std::vector<Place>& valid_places);

  void SetValidPlaces(const std::vector<Place>& valid_places);

  const std::vector<Place>& valid_places() const { return valid_places_; }

 private:
  void UpdateInstNode(Node* in,
                      SSAGraph* graph,
                      Node* inst_node,
                      std::string reshape_output_name);

  std::vector<Place> valid_places_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
