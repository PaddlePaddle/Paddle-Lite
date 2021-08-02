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

#include "lite/model_parser/ssa/op_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

void OpDescBase::UpdateVarBlockIdx(const std::weak_ptr<VarDesc>& var_desc,
                                   int32_t op_block_idx) {
  int32_t init_idx{var_desc.lock()->block_idx()};
  if (init_idx == VarDesc::kInvalidIdx || op_block_idx < init_idx) {
    var_desc.lock()->ResetBlockIdx(op_block_idx);
  }
}

std::set<std::weak_ptr<VarDesc>, VarDescLT> ConvertToSet(
    const std::map<std::string, std::vector<std::weak_ptr<VarDesc>>>& map) {
  std::set<std::weak_ptr<VarDesc>, VarDescLT> set;
  for (const auto& pair : map) {
    set.insert(pair.second.cbegin(), pair.second.cend());
  }
  return set;
}

OpDesc::OpDesc(const general::OpDesc& raw_desc,
               const RootVarScope& scope,
               int32_t block_idx)
    : OpDescBase{raw_desc} {
  for (const auto& param : raw_desc.InputArgumentNames()) {
    for (const auto& var : raw_desc.inputs().at(param)) {
      auto root_var = scope.GetRootVarDesc(var).lock();
      const auto& var_desc = AddInput(param, root_var->latest());
      UpdateVarBlockIdx(var_desc, block_idx);
    }
  }
  for (const auto& param : raw_desc.OutputArgumentNames()) {
    for (const auto& var : raw_desc.outputs().at(param)) {
      auto root_var = scope.GetRootVarDesc(var).lock();
      const auto& var_desc = AddOutput(param, root_var->latest());
      UpdateVarBlockIdx(var_desc, block_idx);
    }
  }
}

std::weak_ptr<VarDesc> OpDesc::AddInput(const std::string& param,
                                        const std::weak_ptr<VarDesc>& desc) {
  auto var_desc = desc.lock()->Read(*this);
  inputs_[param].emplace_back(var_desc);
  return var_desc;
}

std::weak_ptr<VarDesc> OpDesc::AddOutput(const std::string& param,
                                         const std::weak_ptr<VarDesc>& desc) {
  auto var_desc = desc.lock()->Written(*this);
  outputs_[param].emplace_back(var_desc);
  return var_desc;
}

constexpr char WriteBackOp::type_[];
constexpr char WriteBackOp::input_lod_deps_[];
constexpr char WriteBackOp::input_lod_array_deps_[];
constexpr char WriteBackOp::input_src_[];
constexpr char WriteBackOp::input_dst_[];

WriteBackOp::WriteBackOp(const std::weak_ptr<VarDesc>& src,
                         const std::weak_ptr<VarDesc>& dst,
                         int32_t block_idx) {
  CHECK(src.lock()->GetType() == VarDataType::LOD_TENSOR);
  CHECK(dst.lock()->GetType() == VarDataType::LOD_TENSOR);
  AddInput(input_src_, src, block_idx);
  AddInput(input_dst_, dst, block_idx);
  for (auto& op : dst.lock()->target_ops()) {
    for (auto& dep_var : ConvertToSet(op->outputs())) {
      if (dep_var.lock()->GetType() == VarDataType::LOD_TENSOR) {
        AddInput(input_lod_deps_, dep_var, block_idx);
      } else if (dep_var.lock()->GetType() == VarDataType::LOD_TENSOR_ARRAY) {
        AddInput(input_lod_array_deps_, dep_var, block_idx);
      } else {
        LOG(FATAL) << "unsupported dependency var: "
                   << dep_var.lock()->mangled_name();
      }
    }
  }
  fake_desc_.SetType(type_);
}

void WriteBackOp::AddInput(const std::string& param,
                           const std::weak_ptr<VarDesc>& desc,
                           int32_t block_idx) {
  inputs_[param].emplace_back(desc);
  UpdateVarBlockIdx(desc, block_idx);
}

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
