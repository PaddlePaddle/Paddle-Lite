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

#include "driver/huawei_ascend_npu/model_client.h"
#include <memory>
#include <sstream>
#include <string>
#include "driver/huawei_ascend_npu/utility.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

AclModelClient::AclModelClient(int device_id) {
  NNADAPTER_VLOG(5) << "Create a ACL model client(device_id=" << device_id
                    << ")";
  uint32_t device_count = 0;
  ACL_CALL(aclrtGetDeviceCount(&device_count));
  NNADAPTER_VLOG(3) << "device_count: " << device_count;
  NNADAPTER_CHECK_GE(device_id, 0);
  NNADAPTER_CHECK_LT(device_id, device_count);
  InitAclClientEnv(device_id);
}

AclModelClient::~AclModelClient() {
  UnloadModel();
  FinalizeAclClientEnv();
}

void AclModelClient::InitAclClientEnv(int device_id) {
  device_id_ = device_id;
  NNADAPTER_VLOG(5) << "ACL set device(device_id_=" << device_id_ << ")";
  ACL_CALL(aclrtSetDevice(device_id_));
  NNADAPTER_VLOG(5) << "ACL create context";
  ACL_CALL(aclrtCreateContext(&context_, device_id_));
}

void AclModelClient::FinalizeAclClientEnv() {
  NNADAPTER_VLOG(5) << "Destroy ACL context";
  if (context_ != nullptr) {
    ACL_CALL(aclrtDestroyContext(context_));
    context_ = nullptr;
  }
  NNADAPTER_VLOG(5) << "Reset ACL device(device_id_=" << device_id_ << ")";
  ACL_CALL(aclrtResetDevice(device_id_));
}

bool AclModelClient::LoadModel(const void* data, uint32_t size) {
  if (model_desc_) {
    NNADAPTER_LOG(WARNING) << "ACL model had been already loaded.";
    return true;
  }
  ACL_CALL(aclmdlQuerySizeFromMem(
      data, size, &model_memory_size_, &model_weight_size_));
  ACL_CALL(aclrtMalloc(
      &model_memory_ptr_, model_memory_size_, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclrtMalloc(
      &model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclmdlLoadFromMemWithMem(data,
                                    size,
                                    &model_id_,
                                    model_memory_ptr_,
                                    model_memory_size_,
                                    model_weight_ptr_,
                                    model_weight_size_));
  auto model_desc = aclmdlCreateDesc();
  if (!model_desc) {
    NNADAPTER_LOG(ERROR) << "Failed to create ACL model description!";
    return false;
  }
  ACL_CALL(aclmdlGetDesc(model_desc, model_id_));
  model_desc_ = model_desc;
  NNADAPTER_CHECK(CreateModelIODataset());
  NNADAPTER_VLOG(3) << "Load a ACL model success.";
  return true;
}

void AclModelClient::UnloadModel() {
  if (!model_desc_) {
    NNADAPTER_LOG(WARNING) << "No ACL model is loaded.";
    return;
  }
  if (input_dataset_) {
    DestroyDataset(&input_dataset_);
  }
  if (output_dataset_) {
    DestroyDataset(&output_dataset_);
  }
  ACL_CALL(aclmdlUnload(model_id_));
  if (model_memory_ptr_) {
    ACL_CALL(aclrtFree(model_memory_ptr_));
    model_memory_ptr_ = nullptr;
    model_memory_size_ = 0;
  }
  if (model_weight_ptr_) {
    ACL_CALL(aclrtFree(model_weight_ptr_));
    model_weight_ptr_ = nullptr;
    model_weight_size_ = 0;
  }
  ACL_CALL(aclmdlDestroyDesc(model_desc_));
  model_desc_ = nullptr;
  NNADAPTER_VLOG(5) << "Unload a ACL model success(model_id=" << model_id_
                    << ")";
}

bool AclModelClient::GetModelIOTensorDim(
    std::vector<ge::TensorDesc>* input_tensor_descs,
    std::vector<ge::TensorDesc>* output_tensor_descs) {
  if (!model_desc_) {
    NNADAPTER_LOG(FATAL) << "No ACL model is loaded.";
    return false;
  }
  NNADAPTER_CHECK(input_tensor_descs && output_tensor_descs);
  input_tensor_descs->clear();
  output_tensor_descs->clear();
  auto input_count = aclmdlGetNumInputs(model_desc_);
  NNADAPTER_VLOG(3) << "input_count: " << input_count;
  for (size_t i = 0; i < input_count; i++) {
    aclmdlIODims dims;
    ACL_CALL(aclmdlGetInputDims(model_desc_, i, &dims));
    auto data_type = aclmdlGetInputDataType(model_desc_, i);
    auto format = aclmdlGetInputFormat(model_desc_, i);
    std::string name(aclmdlGetInputNameByIndex(model_desc_, i));
    ge::TensorDesc ge_tensor_desc(ge::Shape(ConvertACLDimsToGEDims(dims)),
                                  ConvertACLFormatToGEFormat(format),
                                  ConvertACLDataTypeToGEDataType(data_type));
    ge_tensor_desc.SetName(name.c_str());
    input_tensor_descs->push_back(ge_tensor_desc);
  }
  auto output_count = aclmdlGetNumOutputs(model_desc_);
  NNADAPTER_VLOG(3) << "output_count: " << output_count;
  for (size_t i = 0; i < output_count; i++) {
    aclmdlIODims dims;
    ACL_CALL(aclmdlGetOutputDims(model_desc_, i, &dims));
    auto data_type = aclmdlGetOutputDataType(model_desc_, i);
    auto format = aclmdlGetOutputFormat(model_desc_, i);
    std::string name(aclmdlGetOutputNameByIndex(model_desc_, i));
    ge::TensorDesc ge_tensor_desc(ge::Shape(ConvertACLDimsToGEDims(dims)),
                                  ConvertACLFormatToGEFormat(format),
                                  ConvertACLDataTypeToGEDataType(data_type));
    ge_tensor_desc.SetName(name.c_str());
    output_tensor_descs->push_back(ge_tensor_desc);
  }
  NNADAPTER_VLOG(5)
      << "Get input and output dimensions from a ACL model success.";
  return true;
}

bool AclModelClient::CreateModelIODataset() {
  if (!model_desc_) {
    NNADAPTER_LOG(FATAL) << "No ACL model is loaded.";
    return false;
  }
  if (input_dataset_) {
    DestroyDataset(&input_dataset_);
  }
  input_dataset_ = aclmdlCreateDataset();
  NNADAPTER_CHECK(input_dataset_) << "Failed to create input dataset!";
  auto input_count = aclmdlGetNumInputs(model_desc_);
  NNADAPTER_VLOG(3) << "input_count: " << input_count;
  for (uint32_t i = 0; i < input_count; i++) {
    auto length = aclmdlGetInputSizeByIndex(model_desc_, i);
    NNADAPTER_VLOG(5) << "The buffer length of model input tensor " << i << ":"
                      << length;
    void* device_ptr = nullptr;
    ACL_CALL(aclrtMalloc(&device_ptr, length, ACL_MEM_MALLOC_NORMAL_ONLY));
    auto data_buffer = aclCreateDataBuffer(device_ptr, length);
    NNADAPTER_CHECK(data_buffer)
        << "Failed to call aclCreateDataBuffer to create a data buffer!";
    ACL_CALL(aclmdlAddDatasetBuffer(input_dataset_, data_buffer));
  }
  if (output_dataset_) {
    DestroyDataset(&output_dataset_);
  }
  output_dataset_ = aclmdlCreateDataset();
  NNADAPTER_CHECK(output_dataset_) << "Failed to create output dataset!";
  auto output_count = aclmdlGetNumOutputs(model_desc_);
  NNADAPTER_VLOG(3) << "output_count: " << output_count;
  for (uint32_t i = 0; i < output_count; i++) {
    auto length = aclmdlGetOutputSizeByIndex(model_desc_, i);
    NNADAPTER_VLOG(5) << "The buffer length of model output tensor " << i << ":"
                      << length;
    void* device_ptr = nullptr;
    ACL_CALL(aclrtMalloc(&device_ptr, length, ACL_MEM_MALLOC_NORMAL_ONLY));
    auto data_buffer = aclCreateDataBuffer(device_ptr, length);
    NNADAPTER_CHECK(data_buffer)
        << "Failed to call aclCreateDataBuffer to create a data buffer!";
    ACL_CALL(aclmdlAddDatasetBuffer(output_dataset_, data_buffer));
  }
  NNADAPTER_VLOG(5) << "Create input and output dataset success.";
  return true;
}

void AclModelClient::DestroyDataset(aclmdlDataset** dataset) {
  if (!dataset) {
    NNADAPTER_LOG(WARNING) << "ACL dataset is not initialized!";
    return;
  }
  auto buffer_count = aclmdlGetDatasetNumBuffers(*dataset);
  for (size_t i = 0; i < buffer_count; i++) {
    auto data_buffer = aclmdlGetDatasetBuffer(*dataset, i);
    auto device_ptr = aclGetDataBufferAddr(data_buffer);
    if (device_ptr) {
      ACL_CALL(aclrtFree(device_ptr));
    } else {
      NNADAPTER_LOG(WARNING) << "The device buffer[" << i
                             << "] from a ACL dataset is invalid!";
    }
    ACL_CALL(aclDestroyDataBuffer(data_buffer));
  }
  ACL_CALL(aclmdlDestroyDataset(*dataset));
  *dataset = nullptr;
  NNADAPTER_VLOG(5) << "Destroy a ACL dataset success.";
}

bool AclModelClient::Process(uint32_t input_count,
                             std::vector<NNAdapterOperandType>* input_types,
                             hal::Argument* input_arguments,
                             uint32_t output_count,
                             std::vector<NNAdapterOperandType>* output_types,
                             hal::Argument* output_arguments) {
  if (!model_desc_) {
    NNADAPTER_LOG(FATAL) << "No ACL model is loaded.";
    return false;
  }
  ACL_CALL(aclrtSetCurrentContext(context_));
  auto FindArgumentByIndex = [&](
      hal::Argument* arguments, int index, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
      if (arguments[i].index == index) {
        return &arguments[i];
      }
    }
    return static_cast<hal::Argument*>(nullptr);
  };
  NNADAPTER_CHECK(input_types);
  NNADAPTER_CHECK(input_arguments);
  NNADAPTER_CHECK(output_types);
  NNADAPTER_CHECK(output_arguments);
  NNADAPTER_CHECK_EQ(input_types->size(), input_count);
  NNADAPTER_CHECK_EQ(output_types->size(), output_count);
  NNADAPTER_CHECK(input_dataset_);
  NNADAPTER_CHECK(output_dataset_);
  NNADAPTER_CHECK_EQ(input_count, aclmdlGetDatasetNumBuffers(input_dataset_));
  NNADAPTER_CHECK_EQ(output_count, aclmdlGetDatasetNumBuffers(output_dataset_));
  // Copy the input data from host to device
  for (uint32_t i = 0; i < input_count; i++) {
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    NNADAPTER_CHECK(arg->memory);
    NNADAPTER_CHECK(arg->access);
    auto type = &input_types->at(i);
    auto host_ptr = arg->access(arg->memory, type);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(*type);
    // Query and verify the input dimensions from ACL runtime
    aclmdlIODims dimensions;
    ACL_CALL(aclmdlGetInputDims(model_desc_, i, &dimensions));
    NNADAPTER_CHECK_GE(dimensions.dimCount, type->dimension_count);
    bool dynamic_shape = false;
    for (size_t j = 0; j < dimensions.dimCount; j++) {
      auto& dimension = dimensions.dims[j];
      if (dimension == -1) {
        dimension = type->dimensions[j];
        dynamic_shape = true;
      } else {
        NNADAPTER_CHECK_EQ(dimension, type->dimensions[j])
            << "The " << j << "th dimension of the " << i
            << "th input does not match, expect " << dimension
            << " but recevied " << type->dimensions[j];
      }
    }
    if (dynamic_shape) {
      aclmdlSetInputDynamicDims(model_id_, input_dataset_, i, &dimensions);
    }
    auto data_buffer = aclmdlGetDatasetBuffer(input_dataset_, i);
    auto data_size = aclGetDataBufferSizeV2(data_buffer);
    NNADAPTER_CHECK_LE(length, data_size)
        << "Not enough device memory for the " << i
        << "th input tensor, expect >= " << length << " but recevied "
        << data_size;
    auto device_ptr = aclGetDataBufferAddr(data_buffer);
    NNADAPTER_CHECK(device_ptr);
    ACL_CALL(aclrtMemcpy(
        device_ptr, length, host_ptr, length, ACL_MEMCPY_HOST_TO_DEVICE));
  }
  // Model execution
  auto start_time = GetCurrentUS();
  ACL_CALL(aclmdlExecute(model_id_, input_dataset_, output_dataset_));
  NNADAPTER_VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  // Copy the output data from device to host
  for (uint32_t i = 0; i < output_count; i++) {
    auto arg = FindArgumentByIndex(output_arguments, i, output_count);
    NNADAPTER_CHECK(arg) << "Output argument " << i << " does not exist!";
    NNADAPTER_CHECK(arg->memory);
    NNADAPTER_CHECK(arg->access);
    auto type = &output_types->at(i);
    aclmdlIODims dimensions;
    ACL_CALL(aclmdlGetCurOutputDims(model_desc_, i, &dimensions));
    NNADAPTER_CHECK_EQ(dimensions.dimCount, type->dimension_count);
    ConvertACLDimsToGEDims(
        dimensions, type->dimensions, &type->dimension_count);
    auto host_ptr = arg->access(arg->memory, type);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(*type);
    auto data_buffer = aclmdlGetDatasetBuffer(output_dataset_, i);
    auto data_size = aclGetDataBufferSizeV2(data_buffer);
    NNADAPTER_CHECK_LE(length, data_size)
        << "Not enough device memory for the " << i
        << "th output tensor, expect >= " << length << " but recevied "
        << data_size;
    auto device_ptr = aclGetDataBufferAddr(data_buffer);
    NNADAPTER_CHECK(device_ptr);
    ACL_CALL(aclrtMemcpy(
        host_ptr, length, device_ptr, length, ACL_MEMCPY_DEVICE_TO_HOST));
  }
  NNADAPTER_VLOG(5) << "Process a ACL model success.";
  return true;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
