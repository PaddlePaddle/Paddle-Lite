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
  auto op2pathmap = OpKernelInfoCollector::Global().Getop2path();
  auto kernel2pathmap = OpKernelInfoCollector::Global().Getkernel2path();

  // write used op and kernel info into files
  std::string opf_path = model_dir + "/.ops_list";
  std::string opf_source_path = model_dir + "/.ops_source_list";
  std::string kpf_path = model_dir + "/.kernels_list";
  std::string kpf_source_path = model_dir + "/.kernels_source_list";
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
  CHECK(input_names_.size() > offset)
      << "The network has " << input_names_.size() << " inputs"
      << ", the offset should be less than this.";
  auto *in_var = exec_scope_->FindVar(input_names_[offset]);
  CHECK(in_var) << "no fatch variable " << input_names_[offset]
                << " in exec_scope";
  return in_var->GetMutable<lite::Tensor>();
}

// get inputs names
std::vector<std::string> Predictor::GetInputNames() { return input_names_; }
// get outputnames
std::vector<std::string> Predictor::GetOutputNames() { return output_names_; }
// append the names of inputs and outputs into input_names_ and output_names_
void Predictor::PrepareFeedFetch() {
  auto current_block = program_desc_.GetBlock<cpp::BlockDesc>(0);
  std::vector<cpp::OpDesc *> feeds;
  std::vector<cpp::OpDesc *> fetchs;
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
  const bool model_from_memory = config.model_from_memory();
  LOG(INFO) << "load from memory " << model_from_memory;

  Build(model_path,
        model_file,
        param_file,
        valid_places,
        passes,
        model_type,
        model_from_memory);
}
void Predictor::Build(const std::string &model_path,
                      const std::string &model_file,
                      const std::string &param_file,
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
  Build(program_desc_, valid_places, passes);
}

void Predictor::Build(const cpp::ProgramDesc &desc,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes) {
  program_desc_ = desc;
  std::vector<Place> inner_places = valid_places;
  inner_places.emplace_back(TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny));
  inner_places.emplace_back(
      TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
  Program program(desc, scope_, inner_places);
  /// The first place in valid_places is
  core::KernelPickFactor factor;
  factor.ConsiderTarget();
  factor.ConsiderPrecision();
  factor.ConsiderDataLayout();
  optimizer_.Run(std::move(program), inner_places, factor, passes);
  exec_scope_ = optimizer_.exec_scope();
  PrepareFeedFetch();
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
