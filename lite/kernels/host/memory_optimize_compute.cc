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

#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

class MemoryOptimizeCompute
    : public KernelLite<TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  using param_t = operators::MemoryOptimizeparam;

  void Run() override {
    auto &param = Param<operators::MemoryOptimizeparam>();
    for(size_t i = 0; i < param.memory_reuse_table.size()-1; i++) {
      
      std::string& var_name = param.memory_reuse_table[i];
      std::string& substitute_var_name = param.memory_reuse_table[++i];

      Variable* var = param.exec_scope->FindVar(var_name);
      Variable* substitute_var = param.exec_scope->FindVar(substitute_var_name);
      CHECK(var);
      CHECK(substitute_var);
      var->GetMutable<lite::Tensor>()->ShareDataWith(*(substitute_var->GetMutable<lite::Tensor>()));
    }
  }

  
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    memory_optimize, kHost, kAny, kAny, paddle::lite::kernels::host::MemoryOptimizeCompute, def)
    .Finalize();
