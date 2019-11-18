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

#include "lite/api/light_api.h"
#include <algorithm>

namespace paddle {
namespace lite {

void LightPredictor::Build(const std::string& model_dir,
                           const std::string& model_buffer,
                           const std::string& param_buffer,
                           lite_api::LiteModelType model_type,
                           bool model_from_memory) {
  switch (model_type) {
#ifndef LITE_ON_TINY_PUBLISH
    case lite_api::LiteModelType::kProtobuf:
      LoadModelPb(model_dir, "", "", scope_.get(), &cpp_program_desc_);
      break;
#endif
    case lite_api::LiteModelType::kNaiveBuffer: {
      if (model_from_memory) {
        LoadModelNaiveFromMemory(
            model_buffer, param_buffer, scope_.get(), &cpp_program_desc_);
      } else {
        LoadModelNaive(model_dir, scope_.get(), &cpp_program_desc_);
      }
      break;
    }
    default:
      LOG(FATAL) << "Unknown model type";
  }
  BuildRuntimeProgram(cpp_program_desc_);
  PrepareFeedFetch();
}

Tensor* LightPredictor::GetInput(size_t offset) {
  CHECK(input_names_.size() > offset)
      << "The network has " << input_names_.size() << " inputs"
      << ", the offset should be less than this.";
  auto* in_var = program_->exec_scope()->FindVar(input_names_[offset]);
  CHECK(in_var) << "no fatch variable " << input_names_[offset]
                << " in exec_scope";
  return in_var->GetMutable<lite::Tensor>();
}

// get input by name
Tensor* LightPredictor::GetInputByName(const std::string& name) {
  auto element = std::find(input_names_.begin(), input_names_.end(), name);
  if (element == input_names_.end()) {
    LOG(ERROR) << "Model do not have input named with: [" << name
               << "], model's inputs include:";
    for (int i = 0; i < input_names_.size(); i++) {
      LOG(ERROR) << "[" << input_names_[i] << "]";
    }
    return nullptr;
  } else {
    int position = std::distance(input_names_.begin(), element);
    return GetInput(position);
  }
}

const Tensor* LightPredictor::GetOutput(size_t offset) {
  CHECK(output_names_.size() > offset)
      << "The network has " << output_names_.size() << " outputs"
      << ", the offset should be less than this.";
  auto* out_var = program_->exec_scope()->FindVar(output_names_.at(offset));
  CHECK(out_var) << "no fatch variable " << output_names_.at(offset)
                 << " in exec_scope";
  return out_var->GetMutable<lite::Tensor>();
}
// get inputs names
std::vector<std::string> LightPredictor::GetInputNames() {
  return input_names_;
}
// get outputnames
std::vector<std::string> LightPredictor::GetOutputNames() {
  return output_names_;
}
// append the names of inputs and outputs into input_names_ and output_names_
void LightPredictor::PrepareFeedFetch() {
  auto current_block = cpp_program_desc_.GetBlock<cpp::BlockDesc>(0);
  std::vector<cpp::OpDesc*> feeds;
  std::vector<cpp::OpDesc*> fetchs;
  for (int i = 0; i < current_block->OpsSize(); i++) {
    auto op = current_block->GetOp<cpp::OpDesc>(i);
    if (op->Type() == "feed") {
      feeds.push_back(op);
    } else if (op->Type() == "fetch") {
      fetchs.push_back(op);
    }
  }
  input_names_.resize(feeds.size());
  output_names_.resize(fetchs.size());
  for (int i = 0; i < feeds.size(); i++) {
    input_names_[feeds[i]->GetAttr<int>("col")] =
        feeds[i]->Output("Out").front();
  }
  for (int i = 0; i < fetchs.size(); i++) {
    output_names_[fetchs[i]->GetAttr<int>("col")] =
        fetchs[i]->Input("X").front();
  }
}

void LightPredictor::BuildRuntimeProgram(const cpp::ProgramDesc& prog) {
  std::vector<Instruction> insts;
  // 1. Create op first
  Program program(prog, scope_, {});

  // 2. Create Instructs

  // Create the kernels of the target places, and filter out the specific
  // kernel with the target alias.
  for (auto& op : program.ops()) {
    auto kernel_type = op->op_info()->GetAttr<std::string>(kKernelTypeAttr);
    std::string op_type, alias;
    Place place;
    KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
    auto kernels = op->CreateKernels({place});
    // filter out a kernel
    auto it = std::find_if(
        kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& it) {
          return it->alias() == alias;
        });
    CHECK(it != kernels.end());
    (*it)->SetContext(ContextScheduler::Global().NewContext((*it)->target()));

    insts.emplace_back(op, std::move(*it));
  }
  program_.reset(new RuntimeProgram(std::move(insts)));

  CHECK(program.exec_scope());
  program_->set_exec_scope(program.exec_scope());
}

}  // namespace lite
}  // namespace paddle
