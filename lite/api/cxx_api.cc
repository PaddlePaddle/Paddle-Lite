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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/paddle_use_passes.h"
#include "lite/utils/io.h"

namespace paddle {
namespace lite {

std::vector<std::string> GetAllOps() {
  return OpLiteFactory::Global().GetAllOps();
}

void Predictor::SaveModel(const std::string &dir,
                          lite_api::LiteModelType model_type,
                          bool record_info) {
  if (!program_) {
    GenRuntimeProgram();
  }
  program_->SaveToProgram(program_desc_);
  switch (model_type) {
    case lite_api::LiteModelType::kProtobuf:
      SaveModelPb(dir, *program_->exec_scope(), *program_desc_.get(), true);
      break;
    case lite_api::LiteModelType::kNaiveBuffer:
      SaveModelNaive(dir, *program_->exec_scope(), *program_desc_.get());
      break;
    default:
      LOG(FATAL) << "Unknown model type";
  }
  if (record_info) {
    MkDirRecur(dir);
    SaveOpKernelInfo(dir);
  }
}

void Predictor::SaveOpKernelInfo(const std::string &model_dir) {
  std::set<std::string> ops_info;
  std::set<std::string> kernels_info;
  auto block_size = program_->block_size();
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    const auto &insts = program_->instructions(block_idx);
    for (auto &inst : insts) {
      // parse op type infomation
      auto op = inst.op()->op_info();
      ops_info.insert(op->Type());
      // parse kernel type information
      std::string kernel_type_str =
          inst.kernel()->op_type() + "," + TargetRepr(inst.kernel()->target()) +
          "," + PrecisionRepr(inst.kernel()->precision()) + "," +
          DataLayoutRepr(inst.kernel()->layout()) + "," +
          inst.kernel()->alias();
      kernels_info.insert(kernel_type_str);
    }
  }

  // get souce_file name from op type and kernel type
  auto op2pathmap = OpKernelInfoCollector::Global().GetOp2PathDict();
  auto kernel2pathmap = OpKernelInfoCollector::Global().GetKernel2PathDict();

  // write used op and kernel info into files
  std::string opf_path = model_dir + "/" + TAILORD_OPS_LIST_NAME;
  std::string opf_source_path =
      model_dir + "/" + TAILORD_OPS_SOURCE_LIST_FILENAME;
  std::string kpf_path = model_dir + "/" + TAILORD_KERNELS_LIST_NAME;
  std::string kpf_source_path =
      model_dir + "/" + TAILORD_KERNELS_SOURCE_LIST_FILENAME;
  std::map<std::string, std::string> op2path;

  std::FILE *opf = std::fopen(opf_path.c_str(), "w");
  std::FILE *opf_source = std::fopen(opf_source_path.c_str(), "w");
  std::FILE *kpf = std::fopen(kpf_path.c_str(), "w");
  std::FILE *kpf_source = std::fopen(kpf_source_path.c_str(), "w");
  std::vector<std::string> opcompile;
  std::vector<std::string> kernelcompile;

  if (nullptr == opf || nullptr == opf_source || nullptr == opf ||
      nullptr == kpf_source) {
    LOG(FATAL) << "failed to create info file into: " << model_dir;
  }
  for (auto op_info = ops_info.begin(); op_info != ops_info.end(); op_info++) {
    fputs(op_info->c_str(), opf);
    fputc('\n', opf);
    std::string op_path = op2pathmap[*op_info];
    fputs(op_path.c_str(), opf_source);
    fputc('\n', opf_source);
  }
  std::fclose(opf_source);
  std::fclose(opf);
  LOG(INFO) << "operators information of tailored model is stored into: "
            << opf_path;

  // write Kernel_type and Kernel_path into file
  for (auto kernel_info = kernels_info.begin();
       kernel_info != kernels_info.end();
       kernel_info++) {
    fputs(kernel_info->c_str(), kpf);
    fputc('\n', kpf);
    std::string kernel_path = kernel2pathmap[*kernel_info];
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
  LOG(INFO) << "kernels information of tailored model is stored into: "
            << kpf_path;
}

#ifndef LITE_WITH_FPGA
lite::Tensor *Predictor::GetInput(size_t offset) {
  CHECK(input_names_.size() > offset)
      << "The network has " << input_names_.size() << " inputs"
      << ", the offset should be less than this.";
  auto *in_var = exec_scope_->FindVar(input_names_[offset]);
  CHECK(in_var) << "no fatch variable " << input_names_[offset]
                << " in exec_scope";
  return in_var->GetMutable<lite::Tensor>();
}
#else
lite::Tensor *Predictor::GetInput(size_t offset) {
  auto *_feed_list = exec_scope_->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto *feed_list = _feed_list->GetMutable<std::vector<lite::Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }
  return &feed_list->at(offset);
}
#endif

// get inputs names
std::vector<std::string> Predictor::GetInputNames() { return input_names_; }

// get outputnames
std::vector<std::string> Predictor::GetOutputNames() { return output_names_; }

// get param names
std::vector<std::string> Predictor::GetParamNames() {
  return exec_scope_->AttributeVarNames();
}

// append the names of inputs and outputs into input_names_ and output_names_
void Predictor::PrepareFeedFetch() {
  if (!program_) {
    GenRuntimeProgram();
  }

  std::vector<const cpp::OpDesc *> feeds;
  std::vector<const cpp::OpDesc *> fetchs;
  const auto &insts = program_->instructions(kRootBlockIdx);
  for (auto &inst : insts) {
    const auto &op = inst.op()->op_info();
    if (op->Type() == "feed") {
      feeds.push_back(op);
    } else if (op->Type() == "fetch") {
      fetchs.push_back(op);
    }
  }

  input_names_.resize(feeds.size());
  output_names_.resize(fetchs.size());
  for (size_t i = 0; i < feeds.size(); i++) {
    input_names_[feeds[i]->GetAttr<int>("col")] =
        feeds[i]->Output("Out").front();
  }
  for (size_t i = 0; i < fetchs.size(); i++) {
    output_names_[fetchs[i]->GetAttr<int>("col")] =
        fetchs[i]->Input("X").front();
  }
}

#ifndef LITE_WITH_FPGA

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
#else

const lite::Tensor *Predictor::GetOutput(size_t offset) const {
  auto *_fetch_list = exec_scope_->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}

std::vector<const lite::Tensor *> Predictor::GetOutputs() const {
  auto *_fetch_list = exec_scope_->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();

  std::vector<const lite::Tensor *> outputs;
  for (auto out : fetch_list) {
    outputs.push_back(&out);
  }
  return outputs;
}

#endif

const cpp::ProgramDesc &Predictor::program_desc() const {
  return *program_desc_.get();
}
const RuntimeProgram &Predictor::runtime_program() const { return *program_; }

void Predictor::Build(const lite_api::CxxConfig &config,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      lite_api::LiteModelType model_type) {
  if (config.is_model_from_memory()) {
    LOG(INFO) << "Load model from memory.";
    Build(config.model_dir(),
          config.model_file(),
          config.param_file(),
          valid_places,
          passes,
          model_type,
          config.get_model_buffer());
  } else {
    LOG(INFO) << "Load model from file.";
    Build(config.model_dir(),
          config.model_file(),
          config.param_file(),
          valid_places,
          passes,
          model_type);
  }
}
void Predictor::Build(const std::string &model_path,
                      const std::string &model_file,
                      const std::string &param_file,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      lite_api::LiteModelType model_type,
                      const lite_api::CxxModelBuffer &model_buffer) {
  switch (model_type) {
    case lite_api::LiteModelType::kProtobuf: {
      bool combined_param = false;
      if (!model_buffer.is_empty() ||
          (!model_file.empty() && !param_file.empty())) {
        combined_param = true;
      }
      LoadModelPb(model_path,
                  model_file,
                  param_file,
                  scope_.get(),
                  program_desc_.get(),
                  combined_param,
                  model_buffer);
    } break;
    case lite_api::LiteModelType::kNaiveBuffer:
      CHECK(!model_path.empty())
          << "NaiveBuffer backend only supported combined param";
      LoadModelNaiveFromFile(model_path, scope_.get(), program_desc_.get());
      break;
    default:
      LOG(FATAL) << "Unknown model type";
  }
  Build(program_desc_, valid_places, passes);
}

void Predictor::Build(const std::shared_ptr<cpp::ProgramDesc> &program_desc,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes) {
  program_desc_ = program_desc;
  // `inner_places` is used to optimize passes
  std::vector<Place> inner_places = valid_places;
  for (auto &valid_place : valid_places) {
    if (valid_place.target == TARGET(kOpenCL)) continue;
    inner_places.emplace_back(
        Place(TARGET(kHost), valid_place.precision, valid_place.layout));
  }

  // Analysis whether the modle is quantized.
  // For quantized model, add place(arm, int8) to inner_places
  const std::vector<std::string> quant_dequant_op = {
      "fake_quantize_abs_max",
      "fake_quantize_range_abs_max",
      "fake_quantize_moving_average_abs_max",
      "fake_quantize_dequantize_moving_average_abs_max",
      "fake_dequantize_max_abs",
      "fake_channel_wise_dequantize_max_abs"};
  bool is_quantized_model = false;
  for (size_t i = 0; i < program_desc_->BlocksSize() && !is_quantized_model;
       ++i) {
    auto *block_desc = program_desc_->GetBlock<cpp::BlockDesc>(i);
    for (size_t j = 0; j < block_desc->OpsSize() && !is_quantized_model; ++j) {
      auto *op_desc = block_desc->GetOp<cpp::OpDesc>(j);
      std::string op_type = op_desc->Type();
      if (std::find(quant_dequant_op.begin(),
                    quant_dequant_op.end(),
                    op_type) != quant_dequant_op.end()) {
        is_quantized_model = true;
      }
    }
  }
  if (is_quantized_model) {
    inner_places.insert(inner_places.begin(),
                        Place{TARGET(kARM), PRECISION(kInt8)});
  }

  Program program(program_desc_, scope_, inner_places);
  valid_places_ = inner_places;

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
  CHECK(var) << "no variable named with " << name << " in exec_scope";
  return &var->Get<lite::Tensor>();
}

lite::Tensor *Predictor::GetMutableTensor(const std::string &name) {
  auto *var = exec_scope_->FindVar(name);
  CHECK(var) << "no variable named with " << name << " in exec_scope";
  return var->GetMutable<lite::Tensor>();
}

// get input by name
lite::Tensor *Predictor::GetInputByName(const std::string &name) {
  auto element = std::find(input_names_.begin(), input_names_.end(), name);
  if (element == input_names_.end()) {
    LOG(ERROR) << "Model do not have input named with: [" << name
               << "], model's inputs include:";
    for (size_t i = 0; i < input_names_.size(); i++) {
      LOG(ERROR) << "[" << input_names_[i] << "]";
    }
    return nullptr;
  } else {
    int position = std::distance(input_names_.begin(), element);
    return GetInput(position);
  }
}

// #ifdef LITE_WITH_TRAIN
// void Predictor::FeedVars(const std::vector<framework::Tensor> &tensors) {
//   auto var = scope_->FindVar("feed");
//   auto &feed_list = *(var->GetMutable<std::vector<lite::Tensor>>());
//   feed_list.resize(tensors.size());

//   for (size_t i = 0; i < tensors.size(); ++i)
//     feed_list[i].ShareDataWith(tensors[i]);
// }
// #endif

}  // namespace lite
}  // namespace paddle
