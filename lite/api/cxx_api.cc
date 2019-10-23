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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/utils/io.h"

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
  SaveOpKernelInfo(dir);
}

void Predictor::SaveOpKernelInfo(const std::string &model_dir) {
  std::vector<std::string> op_info;
  std::vector<std::string> kernel_info;

  for (size_t i = 0; i < program_desc_.BlocksSize(); ++i) {
    auto *cpp_block_desc = program_desc_.GetBlock<cpp::BlockDesc>(i);
    for (size_t j = 0; j < cpp_block_desc->OpsSize(); ++j) {
      // parse op type infomation
      auto op = cpp_block_desc->GetOp<cpp::OpDesc>(j);
      op_info.push_back(op->Type());
      auto kernel_type = op->GetAttr<std::string>(kKernelTypeAttr);
      std::vector<std::string> kernel_result;

      // parse kernel type information
      while (!kernel_type.empty()) {
        size_t next_offset = kernel_type.find("/");
        kernel_result.push_back(kernel_type.substr(0, next_offset));
        if (next_offset == std::string::npos) {
          break;
        } else {
          kernel_type = kernel_type.substr(next_offset + 1);
        }
      }

      int target = std::stoi(kernel_result[2]);
      int precision = std::stoi(kernel_result[3]);
      int layout = std::stoi(kernel_result[4]);
      std::string kernel_type_str =
          kernel_result[0] + "," +
          TargetRepr(static_cast<TargetType>(target)).c_str() + "," +
          PrecisionRepr(static_cast<PrecisionType>(precision)).c_str() + "," +
          DataLayoutRepr(static_cast<DataLayoutType>(layout)).c_str() + "," +
          kernel_result[1];
      kernel_info.push_back(kernel_type_str);
    }
  }

  // remove repeated elements
  std::sort(op_info.begin(), op_info.end());
  op_info.erase(std::unique(op_info.begin(), op_info.end()), op_info.end());

  std::sort(kernel_info.begin(), kernel_info.end());
  kernel_info.erase(unique(kernel_info.begin(), kernel_info.end()),
                    kernel_info.end());

  // get souce_file name from op type and kernel type
  auto op2pathmap = CollectedInfo::Global().Getop2path();
  auto kernel2pathmap = CollectedInfo::Global().Getkernel2path();

  // write used op and kernel info into files
  std::string opf_path = model_dir + "/ops_list.txt";
  std::string opf_source_path = model_dir + "/ops_source_list.txt";
  std::string kpf_path = model_dir + "/kernels_list.txt";
  std::string kpf_source_path = model_dir + "/kernels_source_list.txt";
  std::map<std::string, std::string> op2path;

  std::FILE *opf = std::fopen(opf_path.c_str(), "w");
  std::FILE *opf_source = std::fopen(opf_source_path.c_str(), "w");
  std::FILE *kpf = std::fopen(kpf_path.c_str(), "w");
  std::FILE *kpf_source = std::fopen(kpf_source_path.c_str(), "w");

  std::vector<std::string> opcompile;
  std::vector<std::string> kernelcompile;

  if (nullptr == opf || nullptr == opf_source || nullptr == opf ||
      nullptr == kpf_source) {
    LOG(INFO) << "create info file error";
    exit(0);
  } else {
    for (size_t i = 0; i < op_info.size(); ++i) {
      // write OP_type and OP_path into file
      fputs(op_info[i].c_str(), opf);
      fputc('\n', opf);
      std::string op_path = op2pathmap[op_info[i]];
      fputs(op_path.c_str(), opf_source);
      fputc('\n', opf_source);
    }

    std::fclose(opf_source);
    std::fclose(opf);

    // write Kernel_type and Kernel_path into file
    for (size_t j = 0; j < kernel_info.size(); ++j) {
      fputs(kernel_info[j].c_str(), kpf);
      fputc('\n', kpf);
      std::string kernel_path = kernel2pathmap[kernel_info[j]];
      fputs(kernel_path.c_str(), kpf_source);
      fputc('\n', kpf_source);
      if (kernel_path == "conv_compute.cc") {
        fputs(
            "conv_depthwise.cc\nconv_direct.cc\nconv_gemmlike.cc\nconv_"
            "winograd.cc\n",
            kpf_source);
      }
    }
    std::fclose(kpf_source);
    std::fclose(kpf);
  }
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
