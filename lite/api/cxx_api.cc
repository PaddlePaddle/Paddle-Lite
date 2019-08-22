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
#include "lite/npu/npu_helper.h"
#endif

namespace paddle {
namespace lite {

void Predictor::SaveModel(const std::string &dir,
                          lite_api::LiteModelType model_type) {
  if (!program_) {
    GenRuntimeProgram();
  }
  program_->SaveOpInfosToProgram(&program_desc_);
  switch (model_type) {
    case lite_api::LiteModelType::kProtobuf:
      SaveModelPb(dir, *program_->exec_scope(), program_desc_);
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
  auto *_feed_list = exec_scope_->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto *feed_list = _feed_list->GetMutable<std::vector<lite::Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }
  return &feed_list->at(offset);
}

const lite::Tensor *Predictor::GetOutput(size_t offset) const {
  auto *_fetch_list = exec_scope_->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}

const std::vector<lite::Tensor> *Predictor::GetOutputs() const {
  auto *_fetch_list = exec_scope_->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  return &fetch_list;
}

const cpp::ProgramDesc &Predictor::program_desc() const {
  return program_desc_;
}
const RuntimeProgram &Predictor::runtime_program() const { return *program_; }

void Predictor::Build(const std::string &model_path,
                      const Place &prefer_place,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      lite_api::LiteModelType model_type) {
  LOG(INFO) << "Load model from " << model_path;
  switch (model_type) {
    case lite_api::LiteModelType::kProtobuf:
      LoadModelPb(model_path, scope_.get(), &program_desc_);
      break;
    case lite_api::LiteModelType::kNaiveBuffer:
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
  optimizer_.Run(std::move(program), valid_places, factor, passes);
  exec_scope_ = optimizer_.exec_scope();
}

void Predictor::GenRuntimeProgram() {
  program_ = optimizer_.GenRuntimeProgram();
  CHECK_EQ(exec_scope_, program_->exec_scope());
  program_generated_ = true;
}

void Predictor::GenNPURuntimeProgram() {
  program_ = optimizer_.GenNPURuntimeProgram();
  CHECK_EQ(exec_scope_, program_->exec_scope());
  program_generated_ = true;
}

const lite::Tensor *Predictor::GetTensor(const std::string &name) const {
  auto *var = exec_scope_->FindVar(name);
  return &var->Get<lite::Tensor>();
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
