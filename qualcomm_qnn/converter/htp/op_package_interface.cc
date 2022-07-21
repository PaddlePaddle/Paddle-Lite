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

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"

#include "driver/qualcomm_qnn/converter/htp/op_package.h"
#include "driver/qualcomm_qnn/converter/htp/utility.h"

// op initialization
INIT_PACKAGE_OP_DEF()

// optimization initialization
INIT_PACKAGE_OPTIMIZATION_DEF()

// op parameter order initialization
INIT_PACKAGE_PARAM_ORDER_DEF()

/*
 * axis parameter name list
 * optional
 * needs to be global in the package
 * one list per package
 * for listing axis parameter names passed into Qnn_AddNode API
 * HTP backend auto-adjusts values in axis parameters based on HTP backfilling
 * note: HTP backend backfills tensor dimensions to 4 dimensions
 * syntax: LIST_PACKAGE_AXIS_PARAMS(...)
 * e.g. LIST_PACKAGE_AXIS_PARAMS("Axis", "AXIS", "axis")
 */
LIST_PACKAGE_AXIS_PARAMS()

/*
 * per-channel quantized op name list
 * optional
 * needs to be global in the package
 * one list per package
 * for listing op names which support per-channel quantization
 * per-axis quantization info of an op is embeded in axisScaleOffsetEncoding
 *   inside Qnn_Tensor_t types
 * HTP backend only supports per-channel scale ops
 *   i.e. along last dimension, offset is always zero
 * if an op name is marked as having per-channel scale support, and in
 *   QNN_AddNode, at least one input, parameter, or output has
 *   QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET type:
 * then:
 *   HTP backend will pass to op implementation function the following:
 *     output(s), input(s), parameter(s),
 *     outputPerChannelScale(s), inputPerChannelScale(s),
 * paramPerChannelScale(s)
 *
 * optimization rules can be used to remove extra perChannelScale tensors
 *
 * syntax: LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(...)
 * e.g. LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(sg_op1Name, sg_op2Name)
 */
LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

Qnn_ErrorHandle_t Initialize(
    QnnOpPackage_GlobalInfrastructure_t global_infrastructure) {
  if (HtpOpPackage::GetIsInitialized()) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  auto op_pkg = HtpOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  // Register package params
  REGISTER_PACKAGE_OPS()
  REGISTER_PACKAGE_OPTIMIZATIONS()
  REGISTER_PACKAGE_PARAM_ORDERS()
  REGISTER_PACKAGE_AXIS_PARAMS()
  REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()

  op_pkg->SetPackageInfo(THIS_PKG_NAME_STR);
  HtpOpPackage::SetIsInitialized(true);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t GetInfo(const QnnOpPackage_Info_t** info) {
  auto op_pkg = HtpOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  QNN_CHECK_STATUS(op_pkg->GetPackageInfo(info));

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t Terminate() {
  HtpOpPackage::DestroyInstance();

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t LogInitialize(QnnLog_Callback_t callback,
                                QnnLog_Level_t max_log_level) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t LogSetLevel(QnnLog_Level_t max_log_level) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t LogTerminate() { return QNN_SUCCESS; }

/* The following functions in this comment are not required by HTP backend,
 * thus no implementations needed.
 */
Qnn_ErrorHandle_t ValidateOpConfig(Qnn_OpConfig_t op_config) {
  (void)op_config;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t CreateOpImpl(
    QnnOpPackage_GraphInfrastructure_t graph_infrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* op_impl_ptr) {
  (void)graph_infrastructure;
  (void)node;
  (void)op_impl_ptr;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t FreeOpImpl(QnnOpPackage_OpImpl_t op_impl) {
  (void)op_impl;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) Qnn_ErrorHandle_t
HtpCustomOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init = Initialize;
  interface->v1_4.terminate = Terminate;
  interface->v1_4.getInfo = GetInfo;
  interface->v1_4.validateOpConfig = ValidateOpConfig;
  interface->v1_4.createOpImpl = CreateOpImpl;
  interface->v1_4.freeOpImpl = FreeOpImpl;
  interface->v1_4.logInitialize = LogInitialize;
  interface->v1_4.logSetLevel = LogSetLevel;
  interface->v1_4.logTerminate = LogTerminate;
  return QNN_SUCCESS;
}
#ifdef __cplusplus
}
#endif

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
