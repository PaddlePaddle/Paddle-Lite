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

#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

template <class T>
void TensorFromVector(const std::vector<T>& src, lite::Tensor* dst) {
  auto* src_ptr = static_cast<const void*>(src.data());
  auto* dst_ptr = static_cast<void*>(dst->mutable_data<T>());
  auto size = src.size() * sizeof(T);
  std::memcpy(dst_ptr, src_ptr, size);
}

class Eliminator : public FuseBase {
 public:
  void BuildPattern() override {
    auto* assign_value_op = OpNode("assign_value", "assign_value");
    auto* out = VarNode("out")->assert_is_op_output("assign_value", "Out");
    *assign_value_op >> *out;
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* assign_node = matched.at("assign_value");
    auto* scope = assign_node->stmt()->op()->scope();
    auto* op_info = assign_node->stmt()->op()->op_info();
    auto shape = op_info->GetAttr<std::vector<int>>("shape");
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); i++) out_shape.push_back(shape[i]);
    auto dtype = op_info->GetAttr<int>("dtype");
    auto fp32_values = op_info->GetAttr<std::vector<float>>("fp32_values");
    auto int32_values = op_info->GetAttr<std::vector<int>>("int32_values");
    auto* out = matched.at("out");
    auto* out_tensor =
        scope->FindVar(out->arg()->name)->GetMutable<lite::Tensor>();
    out_tensor->Resize(out_shape);
    if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
      TensorFromVector(int32_values, out_tensor);
    } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
      TensorFromVector(fp32_values, out_tensor);
    } else {
      LOG(FATAL) << "Unsupported dtype for assign_value_op:" << dtype;
    }
    GraphSafeRemoveNodes(graph, {matched.at("assign_value")});
  }
};

}  // namespace

class AssignValueEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    Eliminator eliminator;
    eliminator(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(assign_value_eliminate_pass,
                  paddle::lite::mir::AssignValueEliminatePass)
    .BindTargets({TARGET(kAny)});
