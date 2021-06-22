// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/huawei_kirin_npu/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

std::shared_ptr<hiai::AiModelMngerClient> LoadOMModelFromBuffer(
    const std::string& model_name,
    std::vector<char>* model_buffer,
    bool* model_comp,
    int freq_level,
    int framework_type,
    int model_type,
    int device_type) {
  // Create a hiai model manager client to load a HiAI OM model
  auto model_client = std::make_shared<hiai::AiModelMngerClient>();
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    NNADAPTER_LOG(WARNING) << "Init a HiAI model client failed!";
    return nullptr;
  }
  // Check HiAI DDK version
  const char* ddk_version = model_client->GetVersion();
  if (ddk_version) {
    NNADAPTER_VLOG(3) << "HiAI DDK version: " << ddk_version;
  } else {
    NNADAPTER_LOG(WARNING) << "Unable to get HiAI DDK version!";
  }
  // Check model compatibility
  NNADAPTER_VLOG(3) << "freq_level: " << freq_level
                    << " framework_type: " << framework_type
                    << " model_type: " << model_type
                    << " device_type: " << device_type;
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level, framework_type, model_type, device_type);
  model_desc->SetModelBuffer(
      reinterpret_cast<const void*>(model_buffer->data()),
      model_buffer->size());
  if (!*model_comp &&
      model_client->CheckModelCompatibility(*model_desc, *model_comp) !=
          hiai::AI_SUCCESS) {
    *model_comp = false;
    NNADAPTER_VLOG(3)
        << "HiAI OM model is NOT compatiblitiable, set model_comp to "
        << *model_comp;
  } else {
    *model_comp = true;
    NNADAPTER_VLOG(3) << "HiAI OM model is compatiblitiable, set model_comp to "
                      << *model_comp;
  }
  // Rebuild and write the data of the compatible model to the model buffer
  if (!*model_comp) {
    std::shared_ptr<hiai::AiModelBuilder> model_builder =
        std::make_shared<hiai::AiModelBuilder>(model_client);
    hiai::MemBuffer* org_model_buffer = model_builder->InputMemBufferCreate(
        reinterpret_cast<void*>(model_buffer->data()), model_buffer->size());
    if (org_model_buffer) {
      std::vector<hiai::MemBuffer*> org_model_buffers;
      org_model_buffers.push_back(org_model_buffer);
      hiai::MemBuffer* new_model_buffer = model_builder->OutputMemBufferCreate(
          framework_type, org_model_buffers);
      // NNADAPTER_VLOG(3) << "new hiai om model buffer memeory size is " <<
      // new_model_buffer->GetMemBufferSize();
      if (new_model_buffer) {
        uint32_t new_model_size = 0;
        if (model_builder->BuildModel(org_model_buffers,
                                      new_model_buffer,
                                      new_model_size) == hiai::AI_SUCCESS) {
          // Need to change to new_model_size as GetMemBufferSize is not
          // correct.
          model_buffer->resize(new_model_size);
          memcpy(reinterpret_cast<void*>(model_buffer->data()),
                 new_model_buffer->GetMemBufferData(),
                 new_model_size);
          // Reset the model buffer
          model_desc->SetModelBuffer(
              reinterpret_cast<const void*>(model_buffer->data()),
              model_buffer->size());
          NNADAPTER_VLOG(3) << "Rebuild the compatible model done.";
        } else {
          NNADAPTER_LOG(FATAL)
              << "Failed to call BuildModel to rebuild the compatible model!";
        }
        model_builder->MemBufferDestroy(new_model_buffer);
      } else {
        NNADAPTER_LOG(FATAL) << "Failed to call OutputMemBufferCreate for "
                                "storing a new compatiable HiAI OM model!";
      }
      model_builder->MemBufferDestroy(org_model_buffer);
    } else {
      NNADAPTER_LOG(FATAL) << "Failed to call InputMemBufferCreate for writing "
                              "an old compatiable HiAI OM model!";
    }
  }
  // Load the compatible model
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs{
      model_desc};
  if (model_client->Load(model_descs) != hiai::AI_SUCCESS) {
    NNADAPTER_LOG(FATAL)
        << "Failed to call AiModelMngerClient to load a HiAI OM model!";
    return nullptr;
  }
  NNADAPTER_VLOG(3) << "Load successed.";
  return model_client;
}

bool BuildOMModelToBuffer(
    std::vector<ge::Operator>& input_operators,   // NOLINT
    std::vector<ge::Operator>& output_operators,  // NOLINT
    std::vector<char>* model_buffer) {
  // Convert a HiAI IR graph to a HiAI OM model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_operators).SetOutputs(output_operators);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);

  // Build a HiAI OM model and serialize it into a HiAI OM buffer
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_buffer;
  if (!ir_build.CreateModelBuff(om_model, om_buffer)) {
    NNADAPTER_LOG(FATAL)
        << "Failed to call CreateModelBuff for storing a HiAI OM model!";
    return false;
  }
  if (!ir_build.BuildIRModel(om_model, om_buffer)) {
    NNADAPTER_LOG(FATAL) << "Failed to call BuildIRModel for converting a HiAI "
                            "IR graph to a HiAI OM model!";
    ir_build.ReleaseModelBuff(om_buffer);
    return false;
  }
  model_buffer->resize(om_buffer.length);
  NNADAPTER_VLOG(3) << "HiAI OM model size:" << om_buffer.length;
  memcpy(reinterpret_cast<void*>(model_buffer->data()),
         reinterpret_cast<void*>(om_buffer.data),
         om_buffer.length);
  ir_build.ReleaseModelBuff(om_buffer);
  NNADAPTER_VLOG(3) << "Build succeeded.";
  return true;
}

ge::DataType ConvertPrecision(NNAdapterOperandPrecisionCode input_precision) {
  ge::DataType output_precision = ge::DT_FLOAT;
  switch (input_precision) {
    case NNADAPTER_TENSOR_BOOL8:
      output_precision = ge::DT_BOOL;
      break;
    case NNADAPTER_TENSOR_INT8:
      output_precision = ge::DT_INT8;
      break;
    case NNADAPTER_TENSOR_INT16:
      output_precision = ge::DT_INT16;
      break;
    case NNADAPTER_TENSOR_INT32:
      output_precision = ge::DT_INT32;
      break;
    case NNADAPTER_TENSOR_INT64:
      output_precision = ge::DT_INT64;
      break;
    case NNADAPTER_TENSOR_UINT8:
      output_precision = ge::DT_UINT8;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = ge::DT_QUINT8;
      break;
    case NNADAPTER_TENSOR_UINT16:
      output_precision = ge::DT_UINT16;
      break;
    case NNADAPTER_TENSOR_UINT32:
      output_precision = ge::DT_UINT32;
      break;
    case NNADAPTER_TENSOR_UINT64:
      output_precision = ge::DT_UINT64;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = ge::DT_FLOAT16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = ge::DT_FLOAT;
      break;
    case NNADAPTER_TENSOR_FLOAT64:
      output_precision = ge::DT_DOUBLE;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to HiAI ge::DataType !";
      break;
  }
  return output_precision;
}

ge::Format ConvertDataLayout(NNAdapterOperandLayoutCode input_layout) {
  ge::Format output_layout = ge::FORMAT_NCHW;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = ge::FORMAT_NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = ge::FORMAT_NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to HiAI ge::Format !";
      break;
  }
  return output_layout;
}

std::vector<int64_t> ConvertDimensions(int32_t* input_dimensions,
                                       uint32_t input_dimensions_count) {
  std::vector<int64_t> output_dimensions(input_dimensions_count);
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    output_dimensions[i] = input_dimensions[i];
  }
  return output_dimensions;
}

int32_t ConvertFuseCode(int32_t input_fuse_code) {
  int output_act_mode;
  switch (input_fuse_code) {
    case NNADAPTER_FUSED_NONE:
      output_act_mode = -1;
      break;
    case NNADAPTER_FUSED_RELU:
      output_act_mode = 1;
      break;
    case NNADAPTER_FUSED_RELU1:
      output_act_mode = 7;
      break;
    case NNADAPTER_FUSED_RELU6:
      output_act_mode = 14;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert the NNAdapter fuse code("
                           << input_fuse_code
                           << ") to a HiAI activation operator!";
      break;
  }
  return output_act_mode;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
