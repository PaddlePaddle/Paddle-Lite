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
#include "driver/huawei_ascend_npu/engine.h"
#include "driver/huawei_ascend_npu/utility.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

AclModelClient::AclModelClient(int device_id,
                               AscendConfigParams* config_params) {
  NNADAPTER_VLOG(5) << "Create a ACL model client(device_id=" << device_id
                    << ")";
  uint32_t device_count = 0;
  ACL_CALL(aclrtGetDeviceCount(&device_count));
  NNADAPTER_VLOG(3) << "device_count: " << device_count;
  NNADAPTER_CHECK_GE(device_id, 0);
  NNADAPTER_CHECK_LT(device_id, device_count);
  InitAclClientEnv(device_id);
  InitAclProfilingEnv(config_params->profiling_file_path);
}

AclModelClient::~AclModelClient() {
  UnloadModel();
  FinalizeAclClientEnv();
  FinalizeAclProfilingEnv();
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
  if (context_) {
    ACL_CALL(aclrtDestroyContext(context_));
    context_ = nullptr;
  }
  NNADAPTER_VLOG(5) << "Reset ACL device(device_id_=" << device_id_ << ")";
  ACL_CALL(aclrtResetDevice(device_id_));
}

void AclModelClient::InitAclProfilingEnv(
    const std::string& profiling_file_path) {
  if (!profiling_file_path.empty()) {
    const char* aclProfPath = profiling_file_path.c_str();
    ACL_CALL(aclprofInit(aclProfPath, strlen(aclProfPath)));
    config_ = aclprofCreateConfig(reinterpret_cast<uint32_t*>(&device_id_),
                                  1,
                                  ACL_AICORE_ARITHMETIC_UTILIZATION,
                                  nullptr,
                                  ACL_PROF_ACL_API | ACL_PROF_TASK_TIME |
                                      ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU);
  }
}

void AclModelClient::FinalizeAclProfilingEnv() {
  NNADAPTER_VLOG(5) << "Destroy ACL profiling config";
  if (config_) {
    ACL_CALL(aclprofDestroyConfig(config_));
    config_ = nullptr;
    ACL_CALL(aclprofFinalize());
  }
}

bool AclModelClient::LoadModel(const void* data,
                               size_t size,
                               AscendConfigParams* config_params) {
  if (model_desc_) {
    NNADAPTER_LOG(WARNING) << "ACL model had been already loaded.";
    return true;
  }
  ACL_CALL(aclmdlLoadFromMem(data, size, &model_id_));
  auto model_desc = aclmdlCreateDesc();
  if (!model_desc) {
    NNADAPTER_LOG(ERROR) << "Failed to create ACL model description!";
    return false;
  }
  ACL_CALL(aclmdlGetDesc(model_desc, model_id_));
  model_desc_ = model_desc;
  bool is_dynamic_shape_range =
      config_params->enable_dynamic_shape_range == "true";
  if (is_dynamic_shape_range &&
      config_params->initial_buffer_length_of_dynamic_shape_range > 0) {
    SetDynamicShapeRangeInitialBufferLength(
        config_params->initial_buffer_length_of_dynamic_shape_range);
  }
  NNADAPTER_CHECK(CreateModelInputDataset(is_dynamic_shape_range, -1));
  NNADAPTER_CHECK(CreateModelOutputDataset(is_dynamic_shape_range, -1));
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

bool AclModelClient::CreateModelInputDataset(bool is_dynamic_shape_range,
                                             int64_t buffer_length) {
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
    int64_t length;
    if (is_dynamic_shape_range) {
      length = buffer_length > dynamic_shape_range_initial_buffer_length_
                   ? buffer_length
                   : dynamic_shape_range_initial_buffer_length_;
    } else {
      length = aclmdlGetInputSizeByIndex(model_desc_, i);
    }
    NNADAPTER_VLOG(5) << "The buffer length of model input tensor " << i << ":"
                      << length;
    void* device_ptr = nullptr;
    ACL_CALL(aclrtMalloc(&device_ptr, length, ACL_MEM_MALLOC_NORMAL_ONLY));
    auto data_buffer = aclCreateDataBuffer(device_ptr, length);
    NNADAPTER_CHECK(data_buffer)
        << "Failed to call aclCreateDataBuffer to create a data buffer!";
    ACL_CALL(aclmdlAddDatasetBuffer(input_dataset_, data_buffer));
  }
  NNADAPTER_VLOG(5) << "Create input dataset success.";
  return true;
}

bool AclModelClient::CreateModelOutputDataset(bool is_dynamic_shape_range,
                                              int64_t buffer_length) {
  if (!model_desc_) {
    NNADAPTER_LOG(FATAL) << "No ACL model is loaded.";
    return false;
  }
  if (output_dataset_) {
    DestroyDataset(&output_dataset_);
  }
  output_dataset_ = aclmdlCreateDataset();
  NNADAPTER_CHECK(output_dataset_) << "Failed to create output dataset!";
  auto output_count = aclmdlGetNumOutputs(model_desc_);
  NNADAPTER_VLOG(3) << "output_count: " << output_count;
  for (uint32_t i = 0; i < output_count; i++) {
    int64_t length;
    if (is_dynamic_shape_range) {
      length = buffer_length > dynamic_shape_range_initial_buffer_length_
                   ? buffer_length
                   : dynamic_shape_range_initial_buffer_length_;
    } else {
      length = aclmdlGetOutputSizeByIndex(model_desc_, i);
    }
    NNADAPTER_VLOG(5) << "The buffer length of model output tensor " << i << ":"
                      << length;
    void* device_ptr = nullptr;
    ACL_CALL(aclrtMalloc(&device_ptr, length, ACL_MEM_MALLOC_NORMAL_ONLY));
    auto data_buffer = aclCreateDataBuffer(device_ptr, length);
    NNADAPTER_CHECK(data_buffer)
        << "Failed to call aclCreateDataBuffer to create a data buffer!";
    ACL_CALL(aclmdlAddDatasetBuffer(output_dataset_, data_buffer));
  }
  NNADAPTER_VLOG(5) << "Create output dataset success.";
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

void AclModelClient::ProfilingStart() {
  if (config_) {
    aclprofStart(config_);
  }
}

void AclModelClient::ProfilingEnd() {
  if (config_) {
    aclprofStop(config_);
  }
}

bool AclModelClient::Process(uint32_t input_count,
                             std::vector<NNAdapterOperandType>* input_types,
                             core::Argument* input_arguments,
                             uint32_t output_count,
                             std::vector<NNAdapterOperandType>* output_types,
                             core::Argument* output_arguments,
                             DynamicShapeMode dynamic_shape_mode) {
  if (!model_desc_) {
    NNADAPTER_LOG(FATAL) << "No ACL model is loaded.";
    return false;
  }
  ACL_CALL(aclrtSetCurrentContext(context_));
  auto FindArgumentByIndex = [&](
      core::Argument* arguments, int index, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
      if (arguments[i].index == index) {
        return &arguments[i];
      }
    }
    return static_cast<core::Argument*>(nullptr);
  };
  NNADAPTER_CHECK(input_types);
  NNADAPTER_CHECK(input_arguments);
  NNADAPTER_CHECK(output_types);
  NNADAPTER_CHECK(output_arguments);
  NNADAPTER_CHECK_EQ(input_types->size(), input_count);
  NNADAPTER_CHECK_EQ(output_types->size(), output_count);
  NNADAPTER_CHECK(input_dataset_);
  NNADAPTER_CHECK(output_dataset_);
  if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_NONE ||
      dynamic_shape_mode == DYNAMIC_SHAPE_MODE_SHAPE_RANGE) {
    NNADAPTER_CHECK_EQ(input_count, aclmdlGetDatasetNumBuffers(input_dataset_));
  } else {
    NNADAPTER_CHECK_LT(input_count, aclmdlGetDatasetNumBuffers(input_dataset_));
  }
  NNADAPTER_CHECK_EQ(output_count, aclmdlGetDatasetNumBuffers(output_dataset_));
  // Copy the input data from host to device
  bool is_dynamic_dims = false;
  std::vector<int64_t> dynamic_dims;
  for (uint32_t i = 0; i < input_count; i++) {
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    NNADAPTER_CHECK(arg->memory);
    NNADAPTER_CHECK(arg->access);
    auto type = input_types->at(i);
    auto host_ptr = arg->access(arg->memory, &type, nullptr);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(type);
    // Reallocate dataset memory space in dynamic_shape_range mode if the
    // current buffer length exceeds max_length
    if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_SHAPE_RANGE) {
      if (length > dynamic_shape_range_initial_buffer_length_) {
        NNADAPTER_LOG(WARNING) << "Not enough device memory for the " << i
                               << "th input tensor, expect >= " << length
                               << " but recevied "
                               << dynamic_shape_range_initial_buffer_length_;
        NNADAPTER_LOG(WARNING) << "Reallocate dataset memory space...";
        CreateModelInputDataset(true, length);
      }
    }
    // Query and verify the input dimensions from ACL runtime
    aclmdlIODims dimensions;
    ACL_CALL(aclmdlGetInputDims(model_desc_, i, &dimensions));
    NNADAPTER_CHECK_GE(dimensions.dimCount, type.dimensions.count);
    bool is_dynamic_shape = false;
    for (size_t j = 0; j < dimensions.dimCount; j++) {
      auto& dimension = dimensions.dims[j];
      if (dimension == -1) {
        dimension = type.dimensions.data[j];
        is_dynamic_shape = true;
      } else {
        NNADAPTER_CHECK_EQ(dimension, type.dimensions.data[j])
            << "The " << j << "th dimension of the " << i
            << "th input does not match, expect " << dimension
            << " but recevied " << type.dimensions.data[j];
      }
      NNADAPTER_VLOG(3) << "The " << j << "th dimension of the " << i
                        << "th input is " << dimension;
      dynamic_dims.push_back(dimension);
    }
    // Set true dynamic shapes
    if (is_dynamic_shape) {
      if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_BATCH_SIZE) {
        aclmdlSetDynamicBatchSize(
            model_id_, input_dataset_, i, type.dimensions.data[0]);
      } else if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH) {
        aclmdlSetDynamicHWSize(model_id_,
                               input_dataset_,
                               i,
                               type.dimensions.data[2],
                               type.dimensions.data[3]);
      } else if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_N_DIMS) {
        is_dynamic_dims = true;
      } else if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_SHAPE_RANGE) {
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_GREATER_THAN(5, 1, 1)
        auto data_type = aclmdlGetInputDataType(model_desc_, i);
        auto format = aclmdlGetInputFormat(model_desc_, i);
        std::vector<int64_t> input_dims = ConvertACLDimsToGEDims(dimensions);
        auto input_tensor_desc = aclCreateTensorDesc(
            data_type, input_dims.size(), input_dims.data(), format);
        aclmdlSetDatasetTensorDesc(input_dataset_, input_tensor_desc, i);
#else
        NNADAPTER_LOG(FATAL)
            << "The dynamic shape range feature is only supported in CANN "
               "5.1.1 "
               "and above."
            << "If you want to use, please upgrade and recompile the library.";
#endif
      } else {
        NNADAPTER_LOG(FATAL) << "Unsupported dynamic shape mode: "
                             << dynamic_shape_mode;
      }
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
  // Set dynamic dims
  if (is_dynamic_dims) {
    size_t index;
    aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
    aclmdlIODims dimensions;
    dimensions.dimCount = dynamic_dims.size();
    for (size_t i = 0; i < dimensions.dimCount; i++) {
      dimensions.dims[i] = dynamic_dims[i];
    }
    aclmdlSetInputDynamicDims(model_id_, input_dataset_, index, &dimensions);
  }
  // Model execution
  auto start_time = GetCurrentUS();
  ProfilingStart();
  ACL_CALL(aclmdlExecute(model_id_, input_dataset_, output_dataset_));
  ProfilingEnd();
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
    NNADAPTER_CHECK_EQ(dimensions.dimCount, type->dimensions.count);
    ConvertACLDimsToGEDims(
        dimensions, type->dimensions.data, &type->dimensions.count);
    auto host_ptr = arg->access(arg->memory, type, nullptr);
    NNADAPTER_CHECK(host_ptr);
    auto length = GetOperandTypeBufferLength(*type);
    // Reallocate dataset memory space in dynamic_shape_range mode if the
    // current buffer length exceeds max_length
    if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_SHAPE_RANGE) {
      if (length > dynamic_shape_range_initial_buffer_length_) {
        NNADAPTER_LOG(WARNING) << "Not enough device memory for the " << i
                               << "th output tensor, expect >= " << length
                               << " but recevied "
                               << dynamic_shape_range_initial_buffer_length_;
        NNADAPTER_LOG(WARNING) << "Reallocate dataset memory space...";
        CreateModelOutputDataset(true, length);
      }
    }
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
