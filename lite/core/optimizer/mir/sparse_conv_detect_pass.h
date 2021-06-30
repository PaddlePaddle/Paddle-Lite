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

#include <memory>
#include <string>
#include "lite/core/mir/pass.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class SparseConvDetectPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  template <typename T>
  int ComputeSparseZeros(const lite::Tensor* weights, const int num);

  template <typename T>
  int ComputeSparseWeight(const lite::Tensor* w_tensor,
                          const int M,
                          const int K,
                          const int N,
                          const int num_nonzeroes,
                          lite::Tensor* nonzero_output_tensor,
                          lite::Tensor* oc_nonzeros_tensor,
                          lite::Tensor* diffs_tensor);
  // Add attribute that's named with 'attr_name' from op_info
  void CopyAttrFromOpInfo(cpp::OpDesc* op_desc,
                          OpInfo* op_info,
                          const std::string& attr_name);
  // Copy an input scale that's named with 'name' from op_info
  void CopyInputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                OpInfo* op_info,
                                const std::string& name);
  // Copy an output scale that's named with 'name' from op_info
  void CopyOutputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                 OpInfo* op_info,
                                 const std::string& name);

 private:
  float thread_hold_{0.4f};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
