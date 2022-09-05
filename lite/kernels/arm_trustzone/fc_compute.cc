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

#include "lite/kernels/arm_trustzone/fc_compute.h"
#include "lite/kernels/arm_trustzone/tee.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/core/program.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm_trustzone {

template <typename Dtype>
void naive_transpose(const Dtype* din, Dtype* dout, int m, int n) {
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      dout[k++] = din[j * n + i];
    }
  }
}

template <PrecisionType PType>
void fc_trans_weights(const Tensor& tin, Tensor* tout);

template <>
void fc_trans_weights<PRECISION(kFloat)>(const Tensor& tin, Tensor* tout) {
}

template <>
void fc_trans_weights<PRECISION(kInt8)>(const Tensor& tin, Tensor* tout) {
}

template <PrecisionType PType, PrecisionType OutType>
bool check_fc_use_gemm(int m, const std::vector<float>& scale, bool has_bias) {
  return m > 1;
}

template <>
bool check_fc_use_gemm<PRECISION(kInt8), PRECISION(kFloat)>(
    int m, const std::vector<float>& scale, bool has_bias) {
  CHECK_GT(scale.size(), 0) << "Int8 FC param must has weight_scale";
  return m > 1 && scale.size() == 1;
}

template <>
bool check_fc_use_gemm<PRECISION(kInt8), PRECISION(kInt8)>(
    int m, const std::vector<float>& scale, bool has_bias) {
  CHECK_GT(scale.size(), 0) << "Int8 FC param must has weight_scale";
  return m > 1 && scale.size() == 1 && !has_bias;
}

template <PrecisionType PType, PrecisionType OutType>
void FcCompute<PType, OutType>::ReInitWhenNeeded() {
  auto& param = this->template Param<operators::FcParam>();
  auto x_dims = param.input->dims();
  if (last_shape_ == x_dims) {
   return;
  }
  last_shape_ = x_dims;
  auto w_dims = param.w->dims();
  auto& ctx = this->ctx_->template As<ARMTrustZoneContext>();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);
  CHECK_GE(param.output->dims().size(), 2UL);

  m_ = x_dims.Slice(0, param.in_num_col_dims).production();
  k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
  CHECK_EQ(k_, w_dims[0]);
  n_ = w_dims[1];
  CHECK_EQ(k_, static_cast<int>(w_dims[0]));
  flag_gemm_ = check_fc_use_gemm<PType, OutType>(
   m_, param.weight_scale, param.bias != nullptr);
  if (!flag_trans_weights_ && !flag_gemm_) {
   flag_trans_weights_ = true;
  }
}

///  for fp32 kernel
template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

/// for int8 kernel with fp32 output
template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->template Param<operators::FcParam>();
}

/// for int8 kernel with int8 output
template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->template Param<operators::FcParam>();
  if (param.bias) {
    flag_trans_bias_ = true;
  }
}

template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMTrustZoneContext>();

  if (!g_init_tee_context) {
    VLOG(4) << "TEE context is not initialized";
    return;
  }

  VLOG(4) << "invoke fccompute(float, float)";
  VLOG(4) << "Prepare to run fc op in TEE...";
  VLOG(4) << "M " << m_ << " N " << n_ << " K " << k_;

  PT_FcParam pt_fc_param;
  memset(&pt_fc_param, 0, sizeof(pt_fc_param));
  pt_fc_param.input = convert_to_portable_tensor(param.input, PT_DataType::kPTFloat, false);
  pt_fc_param.output = convert_to_portable_tensor(param.output, PT_DataType::kPTFloat, true);

  pt_fc_param.w = convert_to_portable_tensor(param.w, PT_DataType::kPTFloat, false);

  if (param.bias){
    pt_fc_param.bias = convert_to_portable_tensor(param.bias, PT_DataType::kPTFloat, false);
  } else 
    pt_fc_param.bias.bytes = nullptr;

  bool flag_act = false;
  lite_api::ActivationType act;
  if (param.activation_type == "relu") {
   act = lite_api::ActivationType::kRelu;
   flag_act = true;
  }
  pt_fc_param.flag_act = flag_act;
  pt_fc_param.flag_trans_weights = flag_trans_weights_;

  handle_t param_handle = create_tee_param(SupportedOp::Fc, (void*)&pt_fc_param);
  VLOG(4) << "Get handle:" << param_handle;
  if (tee_run(SupportedOp::Fc, param_handle) != 0) {
    VLOG(4) << "TEE run error";
    return;
  }
  VLOG(4) << "TEE run finished, then fetch output tensor from TEE:";
  if (fetch_output_tensor(SupportedOp::Fc, param_handle, pt_fc_param.output) != 0) {
    VLOG(4) << "fetch_output_tensor error";
    return;
  }
  VLOG(4) << "Fetch output tensor finished";
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMTrustZoneContext>();

  VLOG(4) << "invoke fccompute(int8, float)";
  bool flag_relu = false;
  operators::ActivationParam act_param;
  lite_api::ActivationType act;
  act_param.has_active = false;
  if (param.activation_type == "relu") {
    act = lite_api::ActivationType::kRelu;
    flag_relu = true;
  }
  VLOG(4) << "Prepare to run fc op in TEE...";
  VLOG(4) << "M " << m_ << " N " << n_ << " K " << k_;

  PT_FcParam pt_fc_param;
  memset(&pt_fc_param, 0, sizeof(pt_fc_param));
  pt_fc_param.input = convert_to_portable_tensor(param.input, PT_DataType::kPTInt8, false);
  pt_fc_param.output = convert_to_portable_tensor(param.output, PT_DataType::kPTFloat, true);
  pt_fc_param.w = convert_to_portable_tensor(param.w, PT_DataType::kPTInt8, false);

  if (param.bias){
    pt_fc_param.bias = convert_to_portable_tensor(param.bias, PT_DataType::kPTFloat, false);
  } else 
    pt_fc_param.bias.bytes = nullptr;

  pt_fc_param.scale = convert_to_portable_tensor(param.weight_scale.data(), PT_DataType::kPTFloat, 
    new DDimLite(std::vector<DDimLite::value_type>({(int)param.weight_scale.size()})));
  pt_fc_param.input_scale = param.input_scale;
  pt_fc_param.output_scale = param.output_scale;
  pt_fc_param.flag_trans_weights = flag_trans_weights_;
  handle_t param_handle = create_tee_param(SupportedOp::Fc, (void*)&pt_fc_param);
  VLOG(4) << "Get handle:" << param_handle;
  if (tee_run(SupportedOp::Fc, param_handle) != 0) {
    VLOG(4) << "TEE run error";
    return;
  }
  VLOG(4) << "TEE run finished, then fetch output tensor from TEE:";
  if (fetch_output_tensor(SupportedOp::Fc, param_handle, pt_fc_param.output) != 0) {
    VLOG(4) << "fetch_output_tensor error";
    return;
  }
  VLOG(4) << "Fetch output tensor finished";
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto& ctx = this->ctx_->template As<ARMTrustZoneContext>();

  VLOG(4) << "invoke fccompute(int8, int8)";
  bool flag_relu = false;
  operators::ActivationParam act_param;
  act_param.has_active = false;
  lite_api::ActivationType act;
  if (param.activation_type == "relu") {
    flag_relu = true;
    act_param.has_active = true;
    act_param.active_type = lite_api::ActivationType::kRelu;
    act = lite_api::ActivationType::kRelu;
  }

  PT_FcParam pt_fc_param;
  memset(&pt_fc_param, 0, sizeof(pt_fc_param));
  pt_fc_param.input = convert_to_portable_tensor(param.input, PT_DataType::kPTInt8, false);
  pt_fc_param.output = convert_to_portable_tensor(param.output, PT_DataType::kPTInt8, true);

  pt_fc_param.w = convert_to_portable_tensor(param.w, PT_DataType::kPTInt8, false);

  if (param.bias){
    pt_fc_param.bias = convert_to_portable_tensor(param.bias, PT_DataType::kPTFloat, false);
  } else 
    pt_fc_param.bias.bytes = nullptr;

  pt_fc_param.scale = convert_to_portable_tensor(param.weight_scale.data(), PT_DataType::kPTFloat, 
    new DDimLite(std::vector<DDimLite::value_type>({(int)param.weight_scale.size()})));
  pt_fc_param.input_scale = param.input_scale;
  pt_fc_param.output_scale = param.output_scale;
  pt_fc_param.flag_trans_weights = flag_trans_weights_;
  pt_fc_param.flag_act = flag_relu;
  handle_t param_handle = create_tee_param(SupportedOp::Fc, (void*)&pt_fc_param);
  VLOG(4) << "Get handle:" << param_handle;
  if (tee_run(SupportedOp::Fc, param_handle) != 0) {
    VLOG(4) << "TEE run error";
    return;
  }
  VLOG(4) << "TEE run finished, then fetch output tensor from TEE:";
  if (fetch_output_tensor(SupportedOp::Fc, param_handle, pt_fc_param.output) != 0) {
    VLOG(4) << "fetch_output_tensor error";
    return;
  }
  VLOG(4) << "Fetch output tensor finished";
}

}  // namespace arm_trustzone
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm_trustzone::FcCompute<PRECISION(kFloat),
														PRECISION(kFloat)>
    FcCompute_FP32;
typedef paddle::lite::kernels::arm_trustzone::FcCompute<PRECISION(kInt8),
                                              PRECISION(kFloat)>
    FcCompute_int8_fp32;
typedef paddle::lite::kernels::arm_trustzone::FcCompute<PRECISION(kInt8),
                                              PRECISION(kInt8)>
    FcCompute_int8_int8;


REGISTER_LITE_KERNEL(fc, kARMTrustZone, kFloat, kNCHW, FcCompute_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kARMTrustZone, kInt8, kNCHW, FcCompute_int8_int8, int8out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kARMTrustZone, kInt8, kNCHW, FcCompute_int8_fp32, fp32out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kFloat))})
    .Finalize();
