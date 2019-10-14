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

#include "lite/api/cxx_api.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/utils/io.h"
#ifdef LITE_WITH_NPU
#include "lite/backends/npu/npu_helper.h"
#endif

namespace paddle {
namespace lite {

void Predictor::SaveModel(const std::string &dir,
                          lite_api::LiteModelType model_type) {
  if (!program_) {
    GenRuntimeProgram();
  }
  program_->SaveOpInfosToProgram(&program_desc_);
  program_->UpdateVarsOfProgram(&program_desc_);
  switch (model_type) {
    case lite_api::LiteModelType::kProtobuf:
      SaveModelPb(dir, *program_->exec_scope(), program_desc_, true);
      break;
    case lite_api::LiteModelType::kNaiveBuffer:
      SaveModelNaive(dir, *program_->exec_scope(), program_desc_);
      break;
    default:
      LOG(FATAL) << "Unknown model type";
  }
#ifdef LITE_WITH_NPU
  for (auto name : npu::DeviceInfo::Global().AllClientNames()) {
    // the npu offline model is saved in current dir
    // so just copy to dst dir
    CHECK_EQ(
        system(string_format("cp -r %s %s", name.c_str(), dir.c_str()).c_str()),
        0)
        << "Failed copy NPU model to " << dir;
  }
#endif
}

lite::Tensor *Predictor::GetInput(size_t offset) {
  CHECK(input_names_.size() > offset)
      << "The network has " << input_names_.size() << " inputs"
      << ", the offset should be less than this.";
  auto *in_var = exec_scope_->FindVar(input_names_[offset]);
  CHECK(in_var) << "no fatch variable " << input_names_[offset]
                << " in exec_scope";
  return in_var->GetMutable<lite::Tensor>();
}

// get inputs names
std::vector<std::string> Predictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto &item : input_names_) {
    input_names.push_back(item.second);
  }
  return input_names;
}
// get outputnames
std::vector<std::string> Predictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto &item : output_names_) {
    output_names.push_back(item.second);
  }
  return output_names;
}
// append the names of inputs and outputs into input_names_ and output_names_
void Predictor::PrepareFeedFetch() {
  auto current_block = program_desc_.GetBlock<cpp::BlockDesc>(0);
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

const lite::Tensor *Predictor::GetOutput(size_t offset) const {
  CHECK(output_names_.size() > offset)
      << "The network has " << output_names_.size() << " outputs"
      << ", the offset should be less than this.";
  const std::string name = output_names_.at(offset);
  auto *out_var = exec_scope_->FindVar(name);
  CHECK(out_var) << "no fatch variable " << name << " in exec_scope";
  return out_var->GetMutable<lite::Tensor>();
}

std::vector<const lite::Tensor *> Predictor::GetOutputs() const {
  std::vector<const lite::Tensor *> outputs;
  size_t out_size = output_names_.size();
  for (size_t i = 0; i < out_size; i++) {
    const std::string name = output_names_.at(i);
    outputs.push_back(GetTensor(name));
  }
  return outputs;
}

const cpp::ProgramDesc &Predictor::program_desc() const {
  return program_desc_;
}
const RuntimeProgram &Predictor::runtime_program() const { return *program_; }

void Predictor::Build(const lite_api::CxxConfig &config,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      lite_api::LiteModelType model_type) {
  const std::string &model_path = config.model_dir();
  const std::string &model_file = config.model_file();
  const std::string &param_file = config.param_file();
  const Place prefer_place = config.preferred_place();
  const bool model_from_memory = config.model_from_memory();
  LOG(INFO) << "load from memory " << model_from_memory;

  Build(model_path,
        model_file,
        param_file,
        prefer_place,
        valid_places,
        passes,
        model_type,
        model_from_memory);
}
void Predictor::Build(const std::string &model_path,
                      const std::string &model_file,
                      const std::string &param_file,
                      const Place &prefer_place,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      lite_api::LiteModelType model_type,
                      bool model_from_memory) {
  switch (model_type) {
    case lite_api::LiteModelType::kProtobuf: {
      bool combined_param = false;
      if (!model_file.empty() && !param_file.empty()) {
        combined_param = true;
      }
      LoadModelPb(model_path,
                  model_file,
                  param_file,
                  scope_.get(),
                  &program_desc_,
                  combined_param,
                  model_from_memory);
    } break;
    case lite_api::LiteModelType::kNaiveBuffer:
      CHECK(!model_path.empty())
          << "NaiveBuffer backend only supported combined param";
      LoadModelNaive(model_path, scope_.get(), &program_desc_);
      break;
    default:
      LOG(FATAL) << "Unknown model type";
  }
  Build(program_desc_, prefer_place, valid_places, passes);
}

void Predictor::Build(const cpp::ProgramDesc &desc,
                      const Place &prefer_place,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes) {
  program_desc_ = desc;
  Program program(desc, scope_, valid_places);
  optimizer_.KernelPickPreferPlace(prefer_place);
  core::KernelPickFactor factor;
  factor.ConsiderTarget();
  factor.ConsiderPrecision();
  factor.ConsiderDataLayout();
  optimizer_.Run(std::move(program), valid_places, factor, passes);
  exec_scope_ = optimizer_.exec_scope();
}

void Predictor::GenRuntimeProgram() {
  program_ = optimizer_.GenRuntimeProgram();
  CHECK_EQ(exec_scope_, program_->exec_scope());
  program_generated_ = true;
}

const lite::Tensor *Predictor::GetTensor(const std::string &name) const {
  auto *var = exec_scope_->FindVar(name);
  return &var->Get<lite::Tensor>();
}
// get input by name
lite::Tensor *Predictor::GetInputByName(const std::string &name) {
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

#ifdef LITE_WITH_TRAIN
void Predictor::FeedVars(const std::vector<framework::Tensor> &tensors) {
  auto var = scope_->FindVar("feed");
  auto &feed_list = *(var->GetMutable<std::vector<lite::Tensor>>());
  feed_list.resize(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i)
    feed_list[i].ShareDataWith(tensors[i]);
}
#endif

}  // namespace lite
}  // namespace paddle
