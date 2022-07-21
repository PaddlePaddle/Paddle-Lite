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

#pragma once
#include <dlfcn.h>
#include <cstdlib>
#include <memory>
#include <mutex>  //NOLINT
#include <vector>
#include "QnnInterface.h"
#include "core/types.h"
#include "driver/qualcomm_qnn/operation/type.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

// Use which lib at runtime, for example: "cpu", "htp"
#define QUALCOMM_QNN_DEVICE "QUALCOMM_QNN_DEVICE"
#define QUALCOMM_QNN_LOG_LEVEL "QUALCOMM_QNN_LOG_LEVEL"

#define QNN_CHECK(a) NNADAPTER_CHECK_EQ((a), QNN_SUCCESS);

typedef enum {
  kCPU = 0,
  kGPU = 1,
  kHTP = 2,
  kUnk = 3,
} DeviceType;

void LogStdoutCallback(const char* fmt,
                       QnnLog_Level_t level,
                       uint64_t timestamp,
                       va_list argp);

QNN_INTERFACE_VER_TYPE GetQnnInterface(void* lib_backend_handle);

Qnn_DataType_t ConvertToQnnDatatype(
    const NNAdapterOperandPrecisionCode precision);

template <typename T>
Qnn_DataType_t GetQnnDatatype();

int GenQnnTensorID();

std::shared_ptr<Qnn_ClientBuffer_t> CreateQnnClientBuf(void* data,
                                                       uint32_t data_size);

std::shared_ptr<Qnn_QuantizeParams_t> CreateQnnQuantizeParams(
    float* scale, int32_t* zero_point);

template <typename T>
Qnn_DataType_t ConvertPodTypeToQnnDataType(std::vector<T> data,
                                           std::vector<uint32_t> dimensions);

Qnn_Tensor_t CreateQnnTensor(
    Qnn_GraphHandle_t* qnn_graph,
    QNN_INTERFACE_VER_TYPE* qnn_interface,
    uint32_t rank,
    Qnn_DataType_t dataType,
    Qnn_TensorType_t type,
    Qnn_QuantizeParams_t* quantize_params = nullptr,
    uint32_t* maxDimensions = nullptr,
    uint32_t* currentDimensions = nullptr,
    Qnn_ClientBuffer_t* clientBuf = nullptr,
    Qnn_TensorDataFormat_t dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    Qnn_TensorMemType_t memType = QNN_TENSORMEMTYPE_RAW);

std::shared_ptr<Qnn_Tensor_t> CreateQnnTensor(Qnn_GraphHandle_t* qnn_graph,
                                              core::Operand* operand);

}  // namespace qualcomm_qnn
}  // namespace nnadapter
