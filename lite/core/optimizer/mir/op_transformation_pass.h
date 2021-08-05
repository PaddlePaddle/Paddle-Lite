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
#include "lite/core/op_registry.h"
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {
/*
 * Op transformation: We convert some ops into other types to reduce the
 * topology
 * complexity
 *    transformation 1 :  depthwise_conv2d_transpose  -----> conv2d_transpose
 */
class OpTransformationPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  // remapping rules:
  // transformation rule1: depthwise_conv2d_transpose  -----> conv2d_transpose
  void ConvertDepthewiseConv2dTranspose2Conv2dTranspose(mir::Node* node);

 private:
  // Common method
  // Add attribute that's named with 'attr_name' from op_info
  void CopyAttrFromOpInfo(cpp::OpDesc* op_desc,
                          OpInfo* op_info,
                          const std::string& attr_name);
  // Copy all inputs from op_info into opdesc
  void CopyAllInputsFromOpInfo(cpp::OpDesc* op_desc, OpInfo* op_info);
  // Copy all outputs from op_info into opdesc
  void CopyAllOutputsFromOpInfo(cpp::OpDesc* op_desc, OpInfo* op_info);
  // Copy an input scale that's named with 'name' from op_info
  void CopyInputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                OpInfo* op_info,
                                const std::string& name);
  // Copy an output scale that's named with 'name' from op_info
  void CopyOutputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                 OpInfo* op_info,
                                 const std::string& name);
  // Update a mir::node from op_desc
  void UpdateNodeFromOpdesc(mir::Node* node, cpp::OpDesc* op_desc);
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
