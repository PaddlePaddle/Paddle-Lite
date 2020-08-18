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
  BuildRuntimeProgram(program_desc_);
  PrepareFeedFetch();
  program_desc_.reset();
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
#endif
    case lite_api::LiteModelType::kNaiveBuffer: {
      if (model_from_memory) {
        LoadModelNaiveFromMemory(
            model_buffer, param_buffer, scope_.get(), program_desc_.get());
      } else {
        LoadModelNaive(model_dir, scope_.get(), program_desc_.get());
      }
      break;
    }
    default:
      LOG(FATAL) << "Unknown model type";
  }

  DequantizeWeight();
  BuildRuntimeProgram(program_desc_);
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
    for (size_t i = 0; i < input_names_.size(); i++) {
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
  for (size_t i = 0; i < feeds.size(); i++) {
    input_names_[feeds[i]->GetAttr<int>("col")] =
        feeds[i]->Output("Out").front();
  }
  for (size_t i = 0; i < fetchs.size(); i++) {
    output_names_[fetchs[i]->GetAttr<int>("col")] =
        fetchs[i]->Input("X").front();
  }
}

void LightPredictor::BuildRuntimeProgram(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc) {
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
        exe_scope->Var(var_desc->Name());
      } else {
        if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") continue;
        scope_->Var(var_desc->Name());
      }
    }
  }
  // Only extracting the ops and generate the runtime program from the main
  // block desc
  program_.reset(new RuntimeProgram(program_desc, exe_scope, kRootBlockIdx));
}

void LightPredictor::DequantizeWeight() {
  std::shared_ptr<const cpp::ProgramDesc> program_desc = program_desc_;
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
    for (size_t k = 0; k < block->OpsSize(); ++k) {
      auto* op_desc = block->GetOp<cpp::OpDesc>(k);
      if (is_weight_quantized_op(op_desc)) {
        auto input_names = op_desc->input_vars();
        for (auto& input_name : input_names) {
          std::string input_scale_name = input_name + "_quant_scale";
          if (op_desc->HasAttr(input_scale_name)) {  // the input is quantized
            auto input_tensor =
                scope_->FindVar(input_name)->GetMutable<lite::Tensor>();
            tmp_tensor.CopyDataFrom(*input_tensor);
            auto scale_list =
                op_desc->GetAttr<std::vector<float>>(input_scale_name);

            int quantize_weight_bits =
                op_desc->GetAttr<int>("quantize_weight_bits");
            CHECK(quantize_weight_bits == 8 || quantize_weight_bits == 16);
            float* fp_data = input_tensor->mutable_data<float>();

            std::string op_type = op_desc->Type();
            if (op_type == "conv2d" || op_type == "depthwise_conv2d") {
              int64_t ch = input_tensor->dims()[0];
              int64_t offset = input_tensor->numel() / ch;
              CHECK_EQ(scale_list.size(), ch);
              if (quantize_weight_bits == 8) {
                const int8_t* int_data = tmp_tensor.data<int8_t>();
                PROCESS_CONV2D_DATA()
              } else {
                const int16_t* int_data = tmp_tensor.data<int16_t>();
                PROCESS_CONV2D_DATA()
              }
            } else if (op_type == "fc" || op_type == "mul") {
              int64_t chin = input_tensor->dims()[0];
              int64_t chout = input_tensor->dims()[1];
              CHECK_EQ(scale_list.size(), chout);
              if (quantize_weight_bits == 8) {
                const int8_t* int_data = tmp_tensor.data<int8_t>();
                PROCESS_FC_DATA()
              } else {
                const int16_t* int_data = tmp_tensor.data<int16_t>();
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

}  // namespace lite
}  // namespace paddle
