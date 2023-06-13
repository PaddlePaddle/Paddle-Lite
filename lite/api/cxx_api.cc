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
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/type_trans_fp16.h"
#endif

namespace paddle {
namespace lite {

std::vector<std::string> GetAllOps() {
  return OpLiteFactory::Global().GetAllOps();
}

bool IsQuantizedMode(const std::shared_ptr<cpp::ProgramDesc> &program_desc) {
  const std::vector<std::string> quant_dequant_op = {
      "fake_quantize_abs_max",
      "fake_quantize_range_abs_max",
      "fake_quantize_moving_average_abs_max",
      "fake_channel_wise_quantize_abs_max",
      "fake_dequantize_max_abs",
      "fake_channel_wise_dequantize_max_abs",
      "fake_quantize_dequantize_abs_max",
      "fake_quantize_dequantize_moving_average_abs_max",
      "fake_channel_wise_quantize_dequantize_abs_max",
      "quantize_linear",
      "dequantize_linear",
  };
  const std::vector<std::string> dynamic_quant_op = {"lstm", "gru"};
  bool is_quantized_model = false;
  for (size_t i = 0; i < program_desc->BlocksSize() && !is_quantized_model;
       ++i) {
    auto *block_desc = program_desc->GetBlock<cpp::BlockDesc>(i);
    for (size_t j = 0; j < block_desc->OpsSize() && !is_quantized_model; ++j) {
      auto *op_desc = block_desc->GetOp<cpp::OpDesc>(j);
      std::string op_type = op_desc->Type();
      if (std::find(quant_dequant_op.begin(),
                    quant_dequant_op.end(),
                    op_type) != quant_dequant_op.end()) {
        is_quantized_model = true;
#ifdef LITE_WITH_XPU
        if (op_desc->HasAttr("bit_length") &&
            op_desc->GetAttr<int32_t>("bit_length") != 8) {
          return false;
        }
#endif
      }

      if (std::find(dynamic_quant_op.begin(),
                    dynamic_quant_op.end(),
                    op_type) != dynamic_quant_op.end()) {
        if (op_desc->HasAttr("quantization_type")) {
          is_quantized_model = true;
        }
      }
    }
  }
  return is_quantized_model;
}

void Predictor::SaveModel(const std::string &dir,
                          lite_api::LiteModelType model_type,
                          bool record_info) {
  if (!program_) {
    GenRuntimeProgram();
  }
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
    if (op_path == "calib_once_op.cc") {
      fputs("calib_op.cc\n", opf_source);
    }
    if (op_path == "io_copy_once_op.cc") {
      fputs("io_copy_op.cc\n", opf_source);
    }
  }
  std::fclose(opf_source);
  std::fclose(opf);
  OPT_LOG << "operators information of tailored model is stored into: "
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
  }
  std::fclose(kpf_source);
  std::fclose(kpf);
  OPT_LOG << "kernels information of tailored model is stored into: "
          << kpf_path;
}

#if !defined(LITE_WITH_METAL)
lite::Tensor *Predictor::GetInput(size_t offset) {
#ifdef LITE_WITH_XPU
  XPU_CALL(xpu_set_device(reinterpret_cast<lite::XPURunTimeOption *>(
                              target_configs_[TARGET(kXPU)].get())
                              ->xpu_dev_num));
#endif
  CHECK(input_names_.size() > offset)
      << "The network has " << input_names_.size() << " inputs"
      << ", the offset should be less than this.";
  auto *in_var = exec_scope_->FindVar(input_names_[offset]);
  CHECK(in_var) << "no feed variable " << input_names_[offset]
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

// get input tensor precision type
const std::vector<PrecisionType> &Predictor::GetInputPrecisions() const {
  return input_precisions_;
}

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
  input_precisions_.resize(feeds.size());
  for (size_t i = 0; i < feeds.size(); i++) {
    input_names_[feeds[i]->GetAttr<int>("col")] =
        feeds[i]->Output("Out").front();
  }
  for (size_t i = 0; i < fetchs.size(); i++) {
    output_names_[fetchs[i]->GetAttr<int>("col")] =
        fetchs[i]->Input("X").front();
  }
  for (size_t i = 0; i < feeds.size(); i++) {
    input_precisions_[i] = GetInput(i)->precision();
  }
}

#if !defined(LITE_WITH_METAL)
const lite::Tensor *Predictor::GetOutput(size_t offset) const {
  CHECK(output_names_.size() > offset)
      << "The network has " << output_names_.size() << " outputs"
      << ", the offset should be less than this.";
  const std::string name = output_names_.at(offset);
  auto *out_var = exec_scope_->FindVar(name);
  CHECK(out_var) << "no fetch variable " << name << " in exec_scope";
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
  CHECK(_fetch_list) << "no fetch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}

std::vector<const lite::Tensor *> Predictor::GetOutputs() const {
  auto *_fetch_list = exec_scope_->FindVar("fetch");
  CHECK(_fetch_list) << "no fetch variable in exec_scope";
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

#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
void Predictor::WeightFP32ToFP16() {
  std::shared_ptr<const cpp::ProgramDesc> program_desc = program_desc_;
  std::vector<std::string> fp16_ops{"conv2d",
                                    "depthwise_conv2d",
                                    "conv2d_transpose",
                                    "fc",
                                    "mul",
                                    "matmul",
                                    "matmul_v2",
                                    "gru",
                                    "sequence_conv",
                                    "elementwise_add",
                                    "elementwise_sub",
                                    "elementwise_div",
                                    "elementwise_mul",
                                    "prelu"};
  for (size_t i = 0; i < program_desc->BlocksSize(); i++) {
    auto *block = program_desc->GetBlock<cpp::BlockDesc>(i);
    for (size_t k = 0; k < block->OpsSize(); ++k) {
      auto *op_desc = block->GetOp<cpp::OpDesc>(k);
      std::string op_type = op_desc->Type();
      auto iter = std::find(fp16_ops.begin(), fp16_ops.end(), op_type);
      if (iter != fp16_ops.end()) {
        auto input_names = op_desc->input_vars();
        for (auto &input_name : input_names) {
          std::string input_weight_name = input_name + "_fp16";
          if (op_desc->HasAttr(input_weight_name)) {  // the input is fp16
            Tensor tmp_tensor;
            auto input_tensor =
                scope_->FindVar(input_name)->GetMutable<lite::Tensor>();
            if (input_tensor->precision() != PRECISION(kFloat)) continue;
            tmp_tensor.CopyDataFrom(*input_tensor);
            input_tensor->clear();
            input_tensor->set_precision(PRECISION(kFP16));
            float16_t *fp_data = input_tensor->mutable_data<float16_t>();
            const float *in_data = tmp_tensor.data<float>();
            lite::arm::math::fp16::fp32_to_fp16(
                in_data, fp_data, input_tensor->numel());
          }
        }
      }
    }
  }
}
#endif  // ENABLE_ARM_FP16

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
          config,
          config.get_model_buffer());
  } else {
    LOG(INFO) << "Load model from file.";
    Build(config.model_dir(),
          config.model_file(),
          config.param_file(),
          valid_places,
          passes,
          model_type,
          config);
  }
}
void Predictor::Build(const std::string &model_path,
                      const std::string &model_file,
                      const std::string &param_file,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      lite_api::LiteModelType model_type,
                      const lite_api::CxxConfig &config,
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
  Build(program_desc_, valid_places, passes, config);
}

void Predictor::Build(const std::shared_ptr<cpp::ProgramDesc> &program_desc,
                      const std::vector<Place> &valid_places,
                      const std::vector<std::string> &passes,
                      const lite_api::CxxConfig &config) {
  program_desc_ = program_desc;
  // `inner_places` is used to optimize passes
  std::vector<Place> inner_places = valid_places;
  for (auto &valid_place : valid_places) {
    if (valid_place.target == TARGET(kOpenCL)) continue;
    inner_places.emplace_back(
        Place(TARGET(kHost), valid_place.precision, valid_place.layout));
  }

  if (IsQuantizedMode(program_desc_)) {
    for (auto &valid_place : valid_places) {
      if (valid_place.target == TARGET(kARM)) {
        inner_places.insert(inner_places.begin(),
                            Place{TARGET(kARM), PRECISION(kInt8)});
      }
      if (valid_place.target == TARGET(kX86)) {
        inner_places.insert(inner_places.begin(),
                            Place{TARGET(kX86), PRECISION(kInt8)});
      }
    }
  }
  // XPU target must make sure to insert in front of others.
  if (IsQuantizedMode(program_desc_)) {
    for (auto &valid_place : valid_places) {
      if (valid_place.target == TARGET(kXPU)) {
        inner_places.insert(inner_places.begin(),
                            Place{TARGET(kXPU), PRECISION(kInt8)});
      }
    }
  }
  Program program(program_desc_, scope_, inner_places);
  valid_places_ = inner_places;

  core::KernelPickFactor factor;
  factor.ConsiderTarget();
  factor.ConsiderPrecision();
  factor.ConsiderDataLayout();

  exec_scope_ = program.exec_scope();

  program_ = RunDefaultOptimizer(
      std::move(program), inner_places, factor, passes, config);

  if (program_desc->HasVersion())
    program_->set_version(program_desc->Version());

  PrepareFeedFetch();
  // Verify if the ops version of current runtime program is
  // the same with that in models.
  CheckPaddleOpVersions(program_desc);

  // Update the runtime program to program_desc only once
  program_->SaveRuntimProgramIntoProgramDesc(program_desc_);

#ifdef ENABLE_ARM_FP16
  // fp16 Weight convert
  WeightFP32ToFP16();
#endif
}

void Predictor::GenRuntimeProgram() {
  CHECK_EQ(exec_scope_, program_->exec_scope());
  program_generated_ = true;
}

void Predictor::Run() {
  if (!program_generated_) {
    GenRuntimeProgram();
  }
  CheckInputValid();

#ifdef LITE_WITH_XPU
  CHECK(target_configs_.count(TARGET(kXPU)))
      << "XPU runtime option is not initialized!";
  XPULoadRunTimeOptionGuard xpu_load_runtime_option_guard(
      reinterpret_cast<lite::XPURunTimeOption *>(
          target_configs_.at(TARGET(kXPU)).get()));
  std::vector<std::vector<int64_t>> query_shape;
  for (size_t i = 0; i < input_names_.size(); i++) {
    query_shape.push_back(std::vector<int64_t>(GetInput(i)->dims().data()));
  }
  lite::TargetWrapperXPU::MallocL3Cache(query_shape);
#endif

  program_->Run();

#ifdef LITE_WITH_XPU
  lite::TargetWrapperXPU::FreeL3Cache();
#endif
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

// get output by name
const lite::Tensor *Predictor::GetOutputByName(const std::string &name) {
  auto element = std::find(output_names_.begin(), output_names_.end(), name);
  if (element == output_names_.end()) {
    LOG(ERROR) << "Model do not have output named with: [" << name
               << "], model's outputs include:";
    for (size_t i = 0; i < output_names_.size(); i++) {
      LOG(ERROR) << "[" << output_names_[i] << "]";
    }
    return nullptr;
  } else {
    int position = std::distance(output_names_.begin(), element);
    return GetOutput(position);
  }
}

/////////////////////////////////////////////////////////////////////////
// Name: CheckPaddleOpVersions
// Author: DannyIsFunny (github)
// Usage: Compare op versions between inputed fluid model and current
//        kernels registry in opt tool.
// Eg. inputed model: Mobilenet_v1, op `conv2d` with version 2.
//     opt tool: op version of kernel `conv2d` should be no less than 2.
/////////////////////////////////////////////////////////////////////////
void Predictor::CheckPaddleOpVersions(
    const std::shared_ptr<cpp::ProgramDesc> &program_desc) {
  // step1. get all the kernels from current programdesc
  auto block_size = program_desc->BlocksSize();
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    const auto &insts = program_->instructions(block_idx);
    for (auto &inst : insts) {
      // 1.1 each kernel from inputed fluid model.
      const auto &op = inst.op()->op_info();
      std::string op_name = op->Type();
      if (program_desc->HasOpVersionMap()) {
        auto *kernel = inst.kernel();
        // Step2. Compared op versions of inputed model and kernel registry.
        // 2.1 Get op_version_map from inputed fluid model.
        auto *model_op_version =
            program_desc->GetOpVersionMap<general::OpVersionMap>();
        // 2.1 Get op_version versions from kernel registry.
        auto kernel_versions =
            ParamTypeRegistry::Global()
                .GetKernelVersion(kernel->key_with_alias(), kernel->place())
                .OpVersions();
        for (auto iter = kernel_versions.begin(); iter != kernel_versions.end();
             iter++) {
          int32_t model_op_version_index =
              model_op_version->GetOpVersionByName(iter->first);
          // Step3. Compared op version between inputed model and kernel
          // registry.
          if ((model_op_version_index > iter->second) &&
              (model_op_version_index != -1)) {
            VLOG(5) << "Warning: incompatible paddle op version. Kernel ("
                    << kernel->name() << ") requires that op_version("
                    << iter->first << ")==" << iter->second
                    << ". However, the op_version(" << iter->first
                    << ") in this models is " << model_op_version_index
                    << ". It's suggested to use PaddlePaddle and "
                       "Paddle-Lite of the same op_version("
                    << iter->first << ").";
          }
        }
      }
    }
  }
}

bool Predictor::TryShrinkMemory() {
#ifdef LITE_WITH_ARM
  // Clear ArmL3Cache
  lite::DeviceInfo::Global().ClearArmL3Cache();
#endif
  const std::vector<std::string> &local_var_names =
      program_->exec_scope()->LocalVarNames();
  for (auto &var_name : local_var_names) {
    Variable *var = program_->exec_scope()->FindLocalVar(var_name);
    if (var->IsType<lite::Tensor>()) {
      // Clear unpersistable tensors
      auto *tensor = program_->exec_scope()->FindMutableTensor(var_name);
      if (!tensor->persistable()) {
        tensor->clear();
      }
    } else if (var->IsType<std::vector<Tensor>>()) {
      // Clear unpersistable tensor vector
      auto *tensor_array =
          program_->exec_scope()->FindMutableTensorList(var_name);
      for (auto &tensor : *tensor_array) {
        if (!tensor.persistable()) {
          tensor.clear();
        }
      }
    } else {
      continue;
    }
  }
  return true;
}

void Predictor::CheckInputValid() {
  for (size_t idx = 0; idx < input_precisions_.size(); ++idx) {
    if (GetInput(idx)->precision() != input_precisions_[idx]) {
      LOG(WARNING) << " Error input tensor precision type. Input index (" << idx
                   << ") Tensor name (" << input_names_[idx]
                   << ") Require precision type ("
                   << PrecisionToStr(input_precisions_[idx])
                   << ") Input precision type ("
                   << PrecisionToStr(GetInput(idx)->precision()) << ").";
    }
  }
}

void Predictor::ClearTensorArray(
    const std::shared_ptr<const cpp::ProgramDesc> &program_desc) {
  for (size_t blk_idx = 0; blk_idx < program_desc->BlocksSize(); blk_idx++) {
    const cpp::BlockDesc *block =
        program_desc->GetBlock<cpp::BlockDesc>(blk_idx);
    for (size_t var_idx = 0; var_idx < block->VarsSize(); var_idx++) {
      const cpp::VarDesc *var = block->GetVar<cpp::VarDesc>(var_idx);
      CHECK(var);

      auto *var_ptr = program_->exec_scope()->FindVar(var->Name());
      if (var_ptr->IsType<std::vector<Tensor>>() &&
          (var->Name() != "feed" && var->Name() != "fetch")) {
        std::vector<Tensor> *tensor_array_var =
            program_->exec_scope()->FindMutableTensorList(var->Name());
        CHECK(tensor_array_var);
        tensor_array_var->clear();
      }
    }
  }
}

void Predictor::SetTargetConfigs(
    const std::map<TargetType, std::shared_ptr<void>> &target_configs) {
#ifdef LITE_WITH_XPU
  std::shared_ptr<void> runtime_option =
      std::shared_ptr<lite::XPURunTimeOption>(new lite::XPURunTimeOption);
  target_configs_.emplace(TARGET(kXPU), std::move(runtime_option));
  if (target_configs.at(TARGET(kXPU)).get()) {
    reinterpret_cast<lite::XPURunTimeOption *>(
        target_configs_[TARGET(kXPU)].get())
        ->Set(reinterpret_cast<const lite::XPURunTimeOption *>(
            target_configs.at(TARGET(kXPU)).get()));
  }
#endif
}

void Predictor::SetStream(TargetType target, void *stream) {
  if (target == TARGET(kXPU)) {
#ifdef LITE_WITH_XPU
    CHECK(target_configs_.count(TARGET(kXPU)))
        << "XPU runtime option is not initialized!";
    reinterpret_cast<lite::XPURunTimeOption *>(
        target_configs_[TARGET(kXPU)].get())
        ->xpu_stream.SetXPUStream(stream);
#endif
  }
}

}  // namespace lite
}  // namespace paddle
