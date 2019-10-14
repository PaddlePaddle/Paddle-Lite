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
}

Tensor* LightPredictor::GetInput(size_t offset) {
  auto* _feed_list = program_->exec_scope()->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto* feed_list = _feed_list->GetMutable<std::vector<Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }
  return &feed_list->at(offset);
}

// get input by name
Tensor* LightPredictor::GetInputByName(const std::string& name) {
  if (idx2feeds_.find(name) == idx2feeds_.end()) {
    LOG(ERROR) << "Model do not have input named with: [" << name
               << "], model's inputs include:";
    for (int i = 0; i < input_names_.size(); i++) {
      LOG(ERROR) << "[" << input_names_[i] << "]";
    }
    return NULL;
  } else {
    int idx = idx2feeds_[name];
    return GetInput(idx);
  }
}

const Tensor* LightPredictor::GetOutput(size_t offset) {
  auto* _fetch_list = program_->exec_scope()->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto& fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}
// get inputs names
std::vector<std::string> LightPredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto& item : input_names_) {
    input_names.push_back(item.second);
  }
  return input_names;
}
// get outputnames
std::vector<std::string> LightPredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto& item : output_names_) {
    output_names.push_back(item.second);
  }
  return output_names;
}
// append the names of inputs and outputs into input_names_ and output_names_
void LightPredictor::PrepareFeedFetch() {
  auto current_block = cpp_program_desc_.GetBlock<cpp::BlockDesc>(0);
  for (int i = 0; i < current_block->OpsSize(); i++) {
    auto op = current_block->GetOp<cpp::OpDesc>(i);
    if (op->Type() == "feed") {
      int idx = op->GetAttr<int>("col");
      input_names_[idx] = op->Output("Out").front();
      idx2feeds_[op->Output("Out").front()] = idx;
    } else if (op->Type() == "fetch") {
      int idx = op->GetAttr<int>("col");
      output_names_[idx] = op->Input("X").front();
    }
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
