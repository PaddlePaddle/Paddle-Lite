/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>

#include "framework/program/tensor_desc.h"
#include "program_desc.h"

namespace paddle_mobile {
namespace framework {

ProgramDesc::ProgramDesc(PaddleMobile__Framework__Proto__ProgramDesc *desc) {
  for (int i = 0; i < desc->n_blocks; ++i) {
    blocks_.emplace_back(std::make_shared<BlockDesc>(desc->blocks[i]));
  }
}

void ProgramDesc::Description(std::string header) {
#ifdef PADDLE_MOBILE_DEBUG
  if (header.size()) {
    LOG(kLOG_INFO) << header;
  }

  for (int i = 0; i < this->blocks_.size(); ++i) {
    auto block = this->blocks_[i];
    LOG(kLOG_DEBUG) << "block: " << block->ID();
    LOG(kLOG_INFO) << "block ops size: " << block->Ops().size();
    for (int j = 0; j < block->Ops().size(); ++j) {
      auto op = block->Ops()[j];
      LOG(kLOG_DEBUG1) << "op: " << op->Type();
      for (auto &input : op->GetInputs()) {
        LOG(kLOG_DEBUG2) << "input parameter: " << input.first;
        for (auto &n : input.second) {
          LOG(kLOG_DEBUG3) << "argument - " << n;
        }
      }
      for (auto &output : op->GetOutputs()) {
        LOG(kLOG_DEBUG2) << "output parameter: " << output.first;
        for (auto &n : output.second) {
          LOG(kLOG_DEBUG3) << "argument - " << n;
        }
      }
      for (auto &attr : op->GetAttrMap()) {
        LOG(kLOG_DEBUG2) << "attr name:: " << attr.first;
        LOG(kLOG_DEBUG3) << "argument - " << attr.second;
      }
    }

    for (const auto &var_desc : block->Vars()) {
      if (var_desc->Type() == VARTYPE_TYPE_LOD_TENSOR) {
        LOG(kLOG_DEBUG1) << "var name: " << var_desc->Name();

        const TensorDesc &tensor_desc = var_desc->Tensor_desc();

        LOG(kLOG_DEBUG2) << "in var tensor desc dims size: "
                         << tensor_desc.Dims().size();
        for (int l = 0; l < tensor_desc.Dims().size(); ++l) {
          LOG(kLOG_DEBUG3) << "var tensor desc dim " << l
                           << " value: " << tensor_desc.Dims()[l];
        }
      }
    }
  }

  for (const auto &block : this->blocks_) {
  }
#endif
}

std::shared_ptr<BlockDesc> ProgramDesc::Block(size_t idx) {
  return blocks_[idx];
}

}  // namespace framework
}  // namespace paddle_mobile
