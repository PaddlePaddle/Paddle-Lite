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

namespace nnadapter {
namespace qualcomm_qnn {

void LogStdoutCallback(const char *fmt,
                       QnnLog_Level_t level,
                       uint64_t timestamp,
                       va_list argp) {
  const char *levelStr = "";
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

QNN_INTERFACE_VER_TYPE GetQnnInterface(void *lib_backend_handle) {
  // get lib handle
  NNADAPTER_CHECK(lib_backend_handle);
  typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
      const QnnInterface_t **providerList, uint32_t *numProviders);
  auto qnn_interface_get_providers_fn =
      reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
          dlsym(lib_backend_handle, "QnnInterface_getProviders"));
  NNADAPTER_CHECK(qnn_interface_get_providers_fn);

  // get inferface providers
  const QnnInterface_t *infertface_providers{nullptr};
  uint32_t num_providers{0};
  QNN_CHECK(
      qnn_interface_get_providers_fn(&infertface_providers, &num_providers));
  NNADAPTER_CHECK(infertface_providers);
  NNADAPTER_CHECK_GT(num_providers, 0U);

  QNN_INTERFACE_VER_TYPE qnn_inferface =
      infertface_providers[0].QNN_INTERFACE_VER_NAME;
  QNN_CHECK(qnn_inferface.logInitialize(LogStdoutCallback, QNN_LOG_LEVEL_INFO));
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
#undef GET_QNN_DATATYPE

}  // namespace qualcomm_qnn
}  // namespace nnadapter
