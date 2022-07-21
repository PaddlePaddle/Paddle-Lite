// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/qualcomm_qnn/utility.h"
#include <type_traits>

namespace nnadapter {
namespace qualcomm_qnn {

void LogStdoutCallback(const char* fmt,
                       QnnLog_Level_t level,
                       uint64_t timestamp,
                       va_list argp) {
  const char* levelStr = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      levelStr = " ERROR ";
      break;
    case QNN_LOG_LEVEL_WARN:
      levelStr = "WARNING";
      break;
    case QNN_LOG_LEVEL_INFO:
      levelStr = "  INFO ";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      levelStr = " DEBUG ";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      levelStr = "VERBOSE";
      break;
    case QNN_LOG_LEVEL_MAX:
      levelStr = "UNKNOWN";
      break;
  }
  fprintf(stdout, "[%-7s] ", levelStr);
  vfprintf(stdout, fmt, argp);
  fprintf(stdout, "\n");
}

QNN_INTERFACE_VER_TYPE GetQnnInterface(void* lib_backend_handle) {
  // get lib handle
  NNADAPTER_CHECK(lib_backend_handle);
  typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
      const QnnInterface_t** providerList, uint32_t* numProviders);
  auto qnn_interface_get_providers_fn =
      reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
          dlsym(lib_backend_handle, "QnnInterface_getProviders"));
  NNADAPTER_CHECK(qnn_interface_get_providers_fn);

  // get inferface providers
  const QnnInterface_t* infertface_providers{nullptr};
  uint32_t num_providers{0};
  QNN_CHECK(
      qnn_interface_get_providers_fn(&infertface_providers, &num_providers));
  NNADAPTER_CHECK(infertface_providers);
  NNADAPTER_CHECK_GT(num_providers, 0U);

  QNN_INTERFACE_VER_TYPE qnn_inferface =
      infertface_providers[0].QNN_INTERFACE_VER_NAME;
  return qnn_inferface;
}

Qnn_DataType_t ConvertToQnnDatatype(
    const NNAdapterOperandPrecisionCode precision) {
  switch (precision) {
    case NNADAPTER_BOOL8:
      return QNN_DATATYPE_BOOL_8;
    case NNADAPTER_INT8:
      return QNN_DATATYPE_INT_8;
    case NNADAPTER_UINT8:
      return QNN_DATATYPE_UINT_8;
    case NNADAPTER_INT16:
      return QNN_DATATYPE_INT_16;
    case NNADAPTER_UINT16:
      return QNN_DATATYPE_UINT_16;
    case NNADAPTER_INT32:
      return QNN_DATATYPE_INT_32;
    case NNADAPTER_UINT32:
      return QNN_DATATYPE_UINT_32;
    case NNADAPTER_INT64:
      return QNN_DATATYPE_INT_64;
    case NNADAPTER_UINT64:
      return QNN_DATATYPE_UINT_64;
    case NNADAPTER_FLOAT16:
      return QNN_DATATYPE_FLOAT_16;
    case NNADAPTER_FLOAT32:
      return QNN_DATATYPE_FLOAT_32;
    case NNADAPTER_FLOAT64:
      return QNN_DATATYPE_UNDEFINED;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return QNN_DATATYPE_SFIXED_POINT_8;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return QNN_DATATYPE_UFIXED_POINT_8;
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
      return QNN_DATATYPE_SFIXED_POINT_16;
    case NNADAPTER_QUANT_UINT16_ASYMM_PER_LAYER:
      return QNN_DATATYPE_UFIXED_POINT_16;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      return QNN_DATATYPE_SFIXED_POINT_32;
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      return QNN_DATATYPE_UFIXED_POINT_32;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision) << ") to Qnn_DataType_t !";
      return QNN_DATATYPE_UNDEFINED;
  }
}

#define GET_QNN_DATATYPE(type_, q_type_)   \
  template <>                              \
  Qnn_DataType_t GetQnnDatatype<type_>() { \
    return q_type_;                        \
  }
GET_QNN_DATATYPE(float, QNN_DATATYPE_FLOAT_32);
GET_QNN_DATATYPE(uint32_t, QNN_DATATYPE_UINT_32);
GET_QNN_DATATYPE(bool, QNN_DATATYPE_BOOL_8);
GET_QNN_DATATYPE(int32_t, QNN_DATATYPE_INT_32);
#undef GET_QNN_DATATYPE

int GenQnnTensorID() {
  static int tensor_id = 0;
  static std::mutex mtx;
  mtx.lock();
  int id = ++tensor_id;
  mtx.unlock();
  return id;
}

std::shared_ptr<Qnn_QuantizeParams_t> CreateQnnQuantizeParams(
    float* scale, int32_t* zero_point) {
  std::shared_ptr<Qnn_QuantizeParams_t> quantize_params =
      std::make_shared<Qnn_QuantizeParams_t>();
  *quantize_params = QNN_QUANTIZE_PARAMS_INIT;
  if (scale == nullptr || zero_point == nullptr) {
    return quantize_params;
  }
  quantize_params->encodingDefinition = QNN_DEFINITION_DEFINED;
  quantize_params->quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  quantize_params->scaleOffsetEncoding.scale = *scale;
  quantize_params->scaleOffsetEncoding.offset = *zero_point;
  return quantize_params;
}

std::shared_ptr<Qnn_ClientBuffer_t> CreateQnnClientBuf(void* data,
                                                       uint32_t data_size) {
  std::shared_ptr<Qnn_ClientBuffer_t> client_buf =
      std::make_shared<Qnn_ClientBuffer_t>();
  client_buf->data = data;
  client_buf->dataSize = data_size;
  return client_buf;
}

template <typename T>
Qnn_DataType_t ConvertPodTypeToQnnDataType(std::vector<T> data,
                                           std::vector<uint32_t> dimensions) {
  if (dimensions.size() == 0) {
    if (std::is_same<T, float>::value) {
      return QNN_DATATYPE_FLOAT_32;
    } else if (std::is_same<T, int8_t>::value) {
      return QNN_DATATYPE_INT_8;
    } else if (std::is_same<T, int16_t>::value) {
      return QNN_DATATYPE_INT_16;
    } else if (std::is_same<T, int32_t>::value) {
      return QNN_DATATYPE_INT_32;
    } else if (std::is_same<T, int64_t>::value) {
      return QNN_DATATYPE_INT_64;
    } else if (std::is_same<T, uint8_t>::value) {
      return QNN_DATATYPE_UINT_8;
    } else if (std::is_same<T, uint16_t>::value) {
      return QNN_DATATYPE_UINT_16;
    } else if (std::is_same<T, uint32_t>::value) {
      return QNN_DATATYPE_UINT_32;
    } else if (std::is_same<T, uint64_t>::value) {
      return QNN_DATATYPE_UINT_64;
    } else if (std::is_same<T, bool>::value) {
      return QNN_DATATYPE_BOOL_8;
    } else {
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the POD type to Qnn_DataType_t !";
      return QNN_DATATYPE_UNDEFINED;
    }
  }
  if (dimensions.size() > 0) {
    if (std::is_same<T, int8_t>::value) {
      return QNN_DATATYPE_SFIXED_POINT_8;
    } else if (std::is_same<T, int16_t>::value) {
      return QNN_DATATYPE_SFIXED_POINT_16;
    } else if (std::is_same<T, int32_t>::value) {
      return QNN_DATATYPE_SFIXED_POINT_32;
    } else if (std::is_same<T, uint8_t>::value) {
      return QNN_DATATYPE_UFIXED_POINT_8;
    } else if (std::is_same<T, uint16_t>::value) {
      return QNN_DATATYPE_UFIXED_POINT_16;
    } else if (std::is_same<T, uint32_t>::value) {
      return QNN_DATATYPE_UFIXED_POINT_32;
    } else {
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the POD type to Qnn_DataType_t !";
      return QNN_DATATYPE_UNDEFINED;
    }
  }
  return QNN_DATATYPE_UNDEFINED;
}
template Qnn_DataType_t ConvertPodTypeToQnnDataType<int8_t>(
    std::vector<int8_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<int16_t>(
    std::vector<int16_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<int32_t>(
    std::vector<int32_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<int64_t>(
    std::vector<int64_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<uint8_t>(
    std::vector<uint8_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<uint16_t>(
    std::vector<uint16_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<uint32_t>(
    std::vector<uint32_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<uint64_t>(
    std::vector<uint64_t> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<bool>(
    std::vector<bool> data, std::vector<uint32_t> dimensions);
template Qnn_DataType_t ConvertPodTypeToQnnDataType<float>(
    std::vector<float> data, std::vector<uint32_t> dimensions);

Qnn_Tensor_t CreateQnnTensor(Qnn_GraphHandle_t* qnn_graph,
                             QNN_INTERFACE_VER_TYPE* qnn_interface,
                             uint32_t rank,
                             Qnn_DataType_t data_type,
                             Qnn_TensorType_t type,
                             Qnn_QuantizeParams_t* quantize_params,
                             uint32_t* max_dimensions,
                             uint32_t* current_dimensions,
                             Qnn_ClientBuffer_t* client_buf,
                             Qnn_TensorDataFormat_t data_format,
                             Qnn_TensorMemType_t mem_type) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.id = GenQnnTensorID();
  tensor.rank = rank;
  tensor.type = type;
  tensor.dataFormat = data_format;
  tensor.dataType = data_type;
  if (quantize_params) {
    tensor.quantizeParams.encodingDefinition =
        quantize_params->encodingDefinition;
    tensor.quantizeParams.quantizationEncoding =
        quantize_params->quantizationEncoding;
    tensor.quantizeParams.scaleOffsetEncoding.scale =
        quantize_params->scaleOffsetEncoding.scale;
    tensor.quantizeParams.scaleOffsetEncoding.offset =
        quantize_params->scaleOffsetEncoding.offset;
  }

  tensor.maxDimensions = max_dimensions;
  tensor.currentDimensions = current_dimensions;
  tensor.memType = mem_type;

  if (client_buf) {
    tensor.clientBuf.dataSize = client_buf->dataSize;
    tensor.clientBuf.data = client_buf->data;
  }

  QNN_CHECK(qnn_interface->tensorCreateGraphTensor(*qnn_graph, tensor));
  return tensor;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
