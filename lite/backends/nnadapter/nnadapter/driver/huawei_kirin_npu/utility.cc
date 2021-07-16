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
                    << " device_type: " << device_type
                    << " model_name: " << model_name;
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
  NNADAPTER_VLOG(3) << "Load a HiAI OM model success.";
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
  NNADAPTER_VLOG(3) << "Build a HiAI OM model success.";
  return true;
}

const std::string GEDataTypeToString(ge::DataType data_type) {
  static const std::vector<std::string> datatype2strings{"DT_FLOAT=0",
                                                         "DT_FLOAT16=1",
                                                         "DT_INT8=2",
                                                         "DT_INT32=3",
                                                         "DT_UINT8=4",
                                                         "Unknown=5",
                                                         "DT_INT16=6",
                                                         "DT_UINT16=7",
                                                         "DT_UINT32=8",
                                                         "DT_INT64=9",
                                                         "DT_UINT64=10",
                                                         "DT_DOUBLE=11",
                                                         "DT_BOOL=12",
                                                         "DT_STRING=13"};
  auto index = static_cast<int>(data_type);
  NNADAPTER_CHECK_LT(index, datatype2strings.size());
  return datatype2strings[index];
}

const std::string GEFormatToString(ge::Format format) {
  static const std::vector<std::string> format2strings = {
      "FORMAT_NCHW = 0",
      "FORMAT_NHWC = 1",
      "FORMAT_ND = 2",
      "FORMAT_NC1HWC0 = 3",
      "FORMAT_FRACTAL_Z = 4",
      "FORMAT_NC1C0HWPAD = 5",
      "FORMAT_NHWC1C0 = 6",
      "FORMAT_FSR_NCHW = 7",
      "FORMAT_FRACTAL_DECONV = 8",
      "FORMAT_C1HWNC0 = 9",
      "FORMAT_FRACTAL_DECONV_TRANSPOSE = 10",
      "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11",
      "FORMAT_NC1HWC0_C04 = 12",
      "FORMAT_FRACTAL_Z_C04 = 13",
      "FORMAT_CHWN = 14",
      "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15",
      "FORMAT_HWCN = 16",
      "FORMAT_NC1KHKWHWC0 = 17",
      "FORMAT_BN_WEIGHT = 18",
      "FORMAT_FILTER_HWCK = 19",
      "FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20",
      "FORMAT_HASHTABLE_LOOKUP_KEYS = 21",
      "FORMAT_HASHTABLE_LOOKUP_VALUE = 22",
      "FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23",
      "FORMAT_HASHTABLE_LOOKUP_HITS = 24"};
  auto index = static_cast<int>(format);
  NNADAPTER_CHECK_LT(index, format2strings.size());
  return format2strings[index];
}

const std::string GEShapeToString(ge::Shape shape) {
  std::stringstream ss;
  size_t dim_count = shape.GetDimNum();
  if (dim_count == 0) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < dim_count - 1; i++) {
    ss << shape.GetDim(i) << ",";
  }
  ss << shape.GetDim(dim_count - 1);
  ss << "}";
  return ss.str();
}

int64_t ProductionOfGEShape(ge::Shape shape) {
  int64_t production = 1;
  size_t dim_count = shape.GetDimNum();
  for (size_t i = 0; i < dim_count; i++) {
    auto dimension = shape.GetDim(i);
    NNADAPTER_CHECK_GT(dimension, 0);
    production *= dimension;
  }
  return production;
}

template <>
ge::DataType GetGEDataType<float>() {
  return ge::DT_FLOAT;
}

template <>
ge::DataType GetGEDataType<int8_t>() {
  return ge::DT_INT8;
}

template <>
ge::DataType GetGEDataType<int16_t>() {
  return ge::DT_INT16;
}

template <>
ge::DataType GetGEDataType<int32_t>() {
  return ge::DT_INT32;
}

template <>
ge::DataType GetGEDataType<int64_t>() {
  return ge::DT_INT64;
}

template <>
ge::DataType GetGEDataType<bool>() {
  return ge::DT_BOOL;
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
          << ") to ge::DataType !";
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
          << OperandLayoutCodeToString(input_layout) << ") to ge::Format !";
      break;
  }
  return output_layout;
}

std::vector<int64_t> ConvertDimensions(const int32_t* input_dimensions,
                                       uint32_t input_dimensions_count) {
  std::vector<int64_t> output_dimensions;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    output_dimensions.push_back(input_dimensions[i]);
  }
  return output_dimensions;
}

std::vector<int64_t> ConvertDimensions(
    const std::vector<int32_t>& input_dimensions) {
  return ConvertDimensions(&input_dimensions[0], input_dimensions.size());
}

int32_t ConvertFuseCode(int32_t input_fuse_code) {
  int output_act_mode;
  switch (input_fuse_code) {
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
                           << ") to a GE activation operator!";
      break;
  }
  return output_act_mode;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
