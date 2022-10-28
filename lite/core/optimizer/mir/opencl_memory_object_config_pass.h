// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/env.h"
namespace paddle {
namespace lite {
namespace mir {

class OpenCLMemoryObjectConfigPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph> &graph) override;

 private:
  // Parse opencl memory object config file
  std::string ReadMemoryObjectConfigsFromEnv();

  // According config file to get node
  std::set<Node *> GetNodesFromOpenCLOpConfig(
      SSAGraph *graph, const std::string &ocl_op_configs);

  // Update layout to DATALAYOUT(kNCHW)
  void UpdateLayoutToBuffer(Node *x);

  // Update layout to TARGET(kARM) or TARGET(kX86)
  void UpdateTargetToCPU(Node *x, const std::vector<Place> &valid_places);

  // Compatible for PrecisionType.
  bool PrecTypeCompatible(const PrecisionType &p1, const PrecisionType &p2);

  static bool KernelScoreCmp(
      const std::pair<float, std::unique_ptr<KernelBase>> &a,
      const std::pair<float, std::unique_ptr<KernelBase>> &b) {
    return a.first > b.first;
  }

  // Separate OclMemoryObject to opencl_op_config and cpu_op_config
  void SeparateOclMemoryObject(std::string *opencl_op_config,
                               std::string *cpu_op_config,
                               const std::string &ocl_memory_object_configs);

  // Check if the model's ops are opencl image2d supported. If you encounter
  // unsupported
  // ops, change target
  void CorrectArgumentPlace(SSAGraph *graph);

  // According config file to select the correct kernels
  void MemoryObjectConfig(SSAGraph *graph);

 private:
  bool input_shape_default_ = false;
  int image2d_max_width_size_ = 65536;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
