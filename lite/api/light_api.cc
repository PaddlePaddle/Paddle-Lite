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
#include <map>
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {

void LightPredictor::Build(const std::string& lite_model_file,
                           bool model_from_memory) {
  if (model_from_memory) {
    LoadModelNaiveFromMemory(
        lite_model_file, scope_.get(), program_desc_.get());
  } else {
    LoadModelNaiveFromFile(lite_model_file, scope_.get(), program_desc_.get());
  }

  // For weight quantization of post training, load the int8/16 weights
  // for optimized model, and dequant it to fp32.
  DequantizeWeight();
#ifdef ENABLE_ARM_FP16
  // fp16 Weight convert
  WeightFP32ToFP16();
#endif
  BuildRuntimeProgram(program_desc_, use_low_precision_);
  PrepareFeedFetch();
}

void LightPredictor::Build(const std::string& model_dir,
                           const std::string& model_buffer,
                           const std::string& param_buffer,
                           lite_api::LiteModelType model_type,
                           bool model_from_memory) {
  switch (model_type) {
#ifndef LITE_ON_TINY_PUBLISH
    case lite_api::LiteModelType::kProtobuf:
      LoadModelPb(model_dir, "", "", scope_.get(), program_desc_.get());
      break;
    case lite_api::LiteModelType::kNaiveBuffer: {
      if (model_from_memory) {
        LoadModelNaiveFromMemory(
            model_buffer, param_buffer, scope_.get(), program_desc_.get());
      } else {
        LoadModelNaive(model_dir, scope_.get(), program_desc_.get());
      }
      break;
    }
#endif
    default:
      LOG(FATAL) << "Unknown model type";
  }

  DequantizeWeight();

#ifdef ENABLE_ARM_FP16
  // fp16 Weight convert
  WeightFP32ToFP16();
#endif
  BuildRuntimeProgram(program_desc_, use_low_precision_);
  PrepareFeedFetch();
}

#if !defined(LITE_WITH_FPGA) && !defined(LITE_WITH_METAL)
Tensor* LightPredictor::GetInput(size_t offset) {
  CHECK(input_names_.size() > offset)
      << "The network has " << input_names_.size() << " inputs"
      << ", the offset should be less than this.";
  auto* in_var = program_->exec_scope()->FindVar(input_names_[offset]);
  CHECK(in_var) << "no fatch variable " << input_names_[offset]
                << " in exec_scope";
  return in_var->GetMutable<lite::Tensor>();
}
#else
Tensor* LightPredictor::GetInput(size_t offset) {
  auto* _feed_list = program_->exec_scope()->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto* feed_list = _feed_list->GetMutable<std::vector<lite::Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }
  return &feed_list->at(offset);
}
#endif

// get input by name
Tensor* LightPredictor::GetInputByName(const std::string& name) {
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
const lite::Tensor* LightPredictor::GetOutputByName(const std::string& name) {
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

#if !defined(LITE_WITH_METAL)
const Tensor* LightPredictor::GetOutput(size_t offset) {
  CHECK(output_names_.size() > offset)
      << "The network has " << output_names_.size() << " outputs"
      << ", the offset should be less than this.";
  auto* out_var = program_->exec_scope()->FindVar(output_names_.at(offset));
  CHECK(out_var) << "no fatch variable " << output_names_.at(offset)
                 << " in exec_scope";
  return out_var->GetMutable<lite::Tensor>();
}
#else
const lite::Tensor* LightPredictor::GetOutput(size_t offset) {
  auto* _fetch_list = program_->exec_scope()->FindVar("fetch");
  CHECK(_fetch_list) << "no fetch variable in exec_scope";
  auto& fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}
#endif

// get inputs names
std::vector<std::string> LightPredictor::GetInputNames() {
  return input_names_;
}
// get outputnames
std::vector<std::string> LightPredictor::GetOutputNames() {
  return output_names_;
}
// get input tensor precision type
const std::vector<PrecisionType>& LightPredictor::GetInputPrecisions() const {
  return input_precisions_;
}
// append the names of inputs and outputs into input_names_ and output_names_
void LightPredictor::PrepareFeedFetch() {
  std::vector<const cpp::OpDesc*> feeds;
  std::vector<const cpp::OpDesc*> fetchs;
  std::shared_ptr<const cpp::ProgramDesc> program_desc = program_desc_;
  auto main_block = program_desc->GetBlock<cpp::BlockDesc>(kRootBlockIdx);
  auto op_size = main_block->OpsSize();
  for (size_t op_idx = 0; op_idx < op_size; ++op_idx) {
    auto op_desc = main_block->GetOp<cpp::OpDesc>(op_idx);
    if (op_desc->Type() == "feed") {
      feeds.push_back(op_desc);
    } else if (op_desc->Type() == "fetch") {
      fetchs.push_back(op_desc);
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

void LightPredictor::BuildRuntimeProgram(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    bool use_precision_low) {
  auto* exe_scope = &scope_->NewScope();
  // Prepare workspace
  scope_->Var("feed")->GetMutable<std::vector<lite::Tensor>>();
  scope_->Var("fetch")->GetMutable<std::vector<lite::Tensor>>();
  CHECK(program_desc);
  auto block_size = program_desc->BlocksSize();
  CHECK(block_size);
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    auto block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
    auto var_size = block_desc->VarsSize();
    for (size_t var_idx = 0; var_idx < var_size; ++var_idx) {
      auto var_desc = block_desc->GetVar<cpp::VarDesc>(var_idx);
      if (!var_desc->Persistable()) {
        auto* var = exe_scope->Var(var_desc->Name());
        if (var_desc->GetType() == lite::VarDescAPI::Type::LOD_TENSOR) {
          const auto var_data_type =
              ConvertPrecisionType(var_desc->GetDataType());
          auto* tensor = var->GetMutable<lite::Tensor>();
          tensor->set_precision(var_data_type);
        }
      } else {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") continue;
        scope_->Var(var_desc->Name());
      }
    }
    auto op_size = block_desc->OpsSize();
    for (size_t op_idx = 0; op_idx < op_size; ++op_idx) {
      auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
      if (op_desc->Type() == "lod_array_length") bool_clear_tensor_ = true;
    }
  }

#ifdef ENABLE_ARM_FP16
  int low_precision = 1;
  std::string old_op;

  if (lite::DeviceInfo::Global().has_fp16()) {
    for (size_t i = 0; i < program_desc->BlocksSize(); i++) {
      auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(i);
      for (size_t op_idx = 0; op_idx < block_desc->OpsSize(); op_idx++) {
        auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
        CHECK(op_desc);
        std::string op_type = op_desc->Type();

        auto op = LiteOpRegistry::Global().Create(op_type);

        std::unique_ptr<KernelBase> kernel;
        if (op_desc->HasAttr(kKernelTypeAttr)) {
          // Create op and pick up the best kernel according to the
          // kKernelTypeAttr attribute
          auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
          std::string alias;
          Place place;
          KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
          op->Attach(*op_desc, exe_scope);
          if (op_type != "feed" && op_type != "fetch") {
            if (place.precision == PRECISION(kFloat)) {
              place.precision = PRECISION(kFP16);
            } else if (place.precision == PRECISION(kAny)) {
              place.precision = PRECISION(kFP16);
            } else {
              low_precision = 0;
            }
          }
          auto kernels = op->CreateKernels({place});
          if (kernels.size() == 0) {
            low_precision = 0;
          }
        }
        if (old_op == "feed") {
          if (op_type != "conv2d") {
            low_precision = 0;
          }
        }
        old_op = op_type;
      }
    }
  } else {
    low_precision = 0;
  }
  if (low_precision == 1 && use_precision_low) {
    use_low_precision_ = true;
    LOG(INFO) << "Inference with low precision!";
  } else {
    use_low_precision_ = false;
  }
#endif

  // Only extracting the ops and generate the runtime program from the main
  // block desc
  program_.reset(new RuntimeProgram(
      program_desc, exe_scope, kRootBlockIdx, use_low_precision_));
}

void LightPredictor::DequantizeWeight() {
  std::shared_ptr<const cpp::ProgramDesc> program_desc = program_desc_;
  CHECK(program_desc != nullptr);

#define PROCESS_CONV2D_DATA()                                             \
  for (int64_t i = 0; i < ch; ++i) {                                      \
    for (int64_t j = 0; j < offset; ++j) {                                \
      fp_data[i * offset + j] = scale_list[i] * int_data[i * offset + j]; \
    }                                                                     \
  }

#define PROCESS_FC_DATA()                                               \
  for (int64_t i = 0; i < chin; i++) {                                  \
    for (int64_t j = 0; j < chout; j++) {                               \
      fp_data[i * chout + j] = scale_list[j] * int_data[i * chout + j]; \
    }                                                                   \
  }

  auto is_weight_quantized_op = [](const cpp::OpDesc* op_desc) {
    CHECK(op_desc != nullptr);
    bool result = false;
    if (op_desc->HasAttr("quantization_type")) {
      std::string type = op_desc->GetAttr<std::string>("quantization_type");
      result = (type == "post_weight_abs_max") ||
               (type == "post_weight_channel_wise_abs_max");
    } else {
      result = op_desc->HasAttr("quantize_weight_bits");
    }
    return result;
  };
  Tensor tmp_tensor;
  for (size_t i = 0; i < program_desc->BlocksSize(); i++) {
    auto* block = program_desc->GetBlock<cpp::BlockDesc>(i);
    CHECK(block != nullptr);
    for (size_t k = 0; k < block->OpsSize(); ++k) {
      auto* op_desc = block->GetOp<cpp::OpDesc>(k);
      CHECK(op_desc != nullptr);
      if (is_weight_quantized_op(op_desc)) {
        auto input_names = op_desc->input_vars();
        for (auto& input_name : input_names) {
          std::string input_scale_name = input_name + "_quant_scale";
          size_t found = input_name.find("/target_trans");
          std::string input_scale_name_alias = "";
          if (found != std::string::npos) {
            input_scale_name_alias =
                input_name.substr(0, found) + "_quant_scale";
          }
          if (op_desc->HasAttr(input_scale_name) ||
              (!input_scale_name_alias.empty() &&
               op_desc->HasAttr(
                   input_scale_name_alias))) {  // the input is quantized
            if (!input_scale_name_alias.empty()) {
              input_scale_name = input_scale_name_alias;
              input_name = input_name.substr(0, found);
            }
            Variable* scope_var = scope_->FindVar(input_name);
            CHECK(scope_var != nullptr);
            auto input_tensor = scope_var->GetMutable<lite::Tensor>();
            CHECK(input_tensor != nullptr);
            tmp_tensor.CopyDataFrom(*input_tensor);
            auto scale_list =
                op_desc->GetAttr<std::vector<float>>(input_scale_name);

            int quantize_weight_bits =
                op_desc->GetAttr<int>("quantize_weight_bits");
            CHECK(quantize_weight_bits == 8 || quantize_weight_bits == 16);
            float* fp_data = input_tensor->mutable_data<float>();
            CHECK(fp_data != nullptr);

            std::string op_type = op_desc->Type();
            if (op_type == "conv2d" || op_type == "depthwise_conv2d") {
              int64_t ch = input_tensor->dims()[0];
              int64_t offset = input_tensor->numel() / ch;
              CHECK_EQ(scale_list.size(), ch);
              if (quantize_weight_bits == 8) {
                const int8_t* int_data = tmp_tensor.data<int8_t>();
                CHECK(int_data != nullptr);
                PROCESS_CONV2D_DATA()
              } else {
                const int16_t* int_data = tmp_tensor.data<int16_t>();
                CHECK(int_data != nullptr);
                PROCESS_CONV2D_DATA()
              }
            } else if (op_type == "fc" || op_type == "mul" ||
                       op_type == "lookup_table") {
              int64_t chin = input_tensor->dims()[0];
              int64_t chout = input_tensor->numel() / chin;
              if (input_tensor->dims().size() > 1) {
                chout = input_tensor->dims()[1];
              } else {
                // swap
                chout = chin;
                chin = 1;
              }
              CHECK_EQ(scale_list.size(), chout);
              if (quantize_weight_bits == 8) {
                const int8_t* int_data = tmp_tensor.data<int8_t>();
                CHECK(int_data != nullptr);
                PROCESS_FC_DATA()
              } else {
                const int16_t* int_data = tmp_tensor.data<int16_t>();
                CHECK(int_data != nullptr);
                PROCESS_FC_DATA()
              }
            }
          }
        }
      }
    }
  }

#undef PROCESS_CONV2D_DATA
#undef PROCESS_FC_DATA
}

#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
void LightPredictor::WeightFP32ToFP16() {
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
    auto* block = program_desc->GetBlock<cpp::BlockDesc>(i);
    for (size_t k = 0; k < block->OpsSize(); ++k) {
      auto* op_desc = block->GetOp<cpp::OpDesc>(k);
      std::string op_type = op_desc->Type();
      auto iter = std::find(fp16_ops.begin(), fp16_ops.end(), op_type);
      if (iter != fp16_ops.end()) {
        auto input_names = op_desc->input_vars();
        for (auto& input_name : input_names) {
          std::string input_weight_name = input_name + "_fp16";
          if (op_desc->HasAttr(input_weight_name)) {  // the input is fp16
            Tensor tmp_tensor;
            auto input_tensor =
                scope_->FindVar(input_name)->GetMutable<lite::Tensor>();

            if (input_tensor->precision() != PRECISION(kFloat)) continue;

            tmp_tensor.CopyDataFrom(*input_tensor);
            input_tensor->clear();
            input_tensor->set_precision(PRECISION(kFP16));

            float16_t* fp_data = input_tensor->mutable_data<float16_t>();
            const float* in_data = tmp_tensor.data<float>();
            lite::arm::math::fp16::fp32_to_fp16(
                in_data, fp_data, input_tensor->numel());
          }
        }
      }
    }
  }
}
#endif

void LightPredictor::CheckInputValid() {
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

bool LightPredictor::TryShrinkMemory() {
#ifdef LITE_WITH_ARM
  // Clear ArmL3Cache
  lite::DeviceInfo::Global().ClearArmL3Cache();
#endif
  const std::vector<std::string>& local_var_names =
      program_->exec_scope()->LocalVarNames();
  for (auto& var_name : local_var_names) {
    Variable* var = program_->exec_scope()->FindLocalVar(var_name);
    if (var->IsType<lite::Tensor>()) {
      // Clear unpersistable tensors
      auto* tensor = program_->exec_scope()->FindMutableTensor(var_name);
      if (!tensor->persistable()) {
        tensor->clear();
      }
    } else if (var->IsType<std::vector<Tensor>>()) {
      // Clear unpersistable tensor vector
      auto* tensor_array =
          program_->exec_scope()->FindMutableTensorList(var_name);
      for (auto& tensor : *tensor_array) {
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
void LightPredictor::ClearTensorArray(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc) {
  for (size_t blk_idx = 0; blk_idx < program_desc->BlocksSize(); blk_idx++) {
    const cpp::BlockDesc* block =
        program_desc->GetBlock<cpp::BlockDesc>(blk_idx);
    for (size_t var_idx = 0; var_idx < block->VarsSize(); var_idx++) {
      const cpp::VarDesc* var = block->GetVar<cpp::VarDesc>(var_idx);
      CHECK(var);

      auto* var_ptr = program_->exec_scope()->FindVar(var->Name());
      if (var_ptr->IsType<std::vector<Tensor>>() &&
          (var->Name() != "feed" && var->Name() != "fetch")) {
        std::vector<Tensor>* tensor_array_var =
            program_->exec_scope()->FindMutableTensorList(var->Name());
        CHECK(tensor_array_var);
        tensor_array_var->clear();
      }
    }
  }
}
}  // namespace lite
}  // namespace paddle
