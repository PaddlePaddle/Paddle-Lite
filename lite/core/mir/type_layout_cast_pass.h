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
#include <string>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class TypeLayoutTransformPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  void ComplementInputs(SSAGraph* graph, Node* inst_node, Node* in);

  void AddLayoutInst(const Type& from,
                     const Type& to,
                     Node* in,
                     SSAGraph* graph,
                     Node* inst_node,
                     const std::vector<Place>& valid_places);

  void SetValidPlaces(const std::vector<Place>& valid_places);

  const std::vector<Place>& valid_places() const { return valid_places_; }

 private:
  std::vector<Place> valid_places_;
};

// add preprocess and postprocess attribute for layout op
class OpenCLTypeLayoutTransformPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  std::vector<Place> valid_places_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
