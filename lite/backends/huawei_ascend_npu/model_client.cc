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

#include "lite/backends/huawei_ascend_npu/model_client.h"

namespace paddle {
namespace lite {
namespace huawei_ascend_npu {

bool AclModelClient::LoadFromMem(const void* data, uint32_t size) {
  if (load_flag_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] model is already loaded!";
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

  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] create model description failed!";
    return false;
  }
  ACL_CALL(aclmdlGetDesc(model_desc_, model_id_));

  VLOG(3) << "[HUAWEI_ASCEND_NPU] Load model form memeory success.";
  load_flag_ = true;
  return true;
}

bool AclModelClient::LoadFromFile(const char* model_path) {
  if (load_flag_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] model is already loaded!";
    return true;
  }

  ACL_CALL(
      aclmdlQuerySize(model_path, &model_memory_size_, &model_weight_size_));
  ACL_CALL(aclrtMalloc(
      &model_memory_ptr_, model_memory_size_, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclrtMalloc(
      &model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclmdlLoadFromFileWithMem(model_path,
                                     &model_id_,
                                     model_memory_ptr_,
                                     model_memory_size_,
                                     model_weight_ptr_,
                                     model_weight_size_));

  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] create model description failed!";
    return false;
  }
  ACL_CALL(aclmdlGetDesc(model_desc_, model_id_));

  VLOG(3) << "[HUAWEI_ASCEND_NPU] Load model form file success: " << model_path;
  load_flag_ = true;
  return true;
}

bool AclModelClient::GetModelIOTensorDim(
    std::vector<TensorDesc>* input_tensor,
    std::vector<TensorDesc>* output_tensor) {
  if (!model_desc_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] GetModelIOTensorDim failed!";
    return false;
  }
  size_t input_num = aclmdlGetNumInputs(model_desc_);
  VLOG(3) << "[HUAWEI_ASCEND_NPU] input number is " << input_num;
  for (size_t i = 0; i < input_num; i++) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] printing input [" << i << "] ....";
    aclmdlIODims input_dim;
    ACL_CALL(aclmdlGetInputDims(model_desc_, i, &input_dim));
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    aclFormat data_format = aclmdlGetInputFormat(model_desc_, i);
    TensorDesc tensor_desc = TensorDesc(data_type, input_dim, data_format);
    input_tensor->push_back(tensor_desc);
  }

  size_t output_num = aclmdlGetNumOutputs(model_desc_);
  VLOG(3) << "[HUAWEI_ASCEND_NPU] output number is " << output_num;
  for (size_t i = 0; i < output_num; i++) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] printing output [" << i << "] ....";
    aclmdlIODims output_dim;
    ACL_CALL(aclmdlGetOutputDims(model_desc_, i, &output_dim));
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    aclFormat data_format = aclmdlGetOutputFormat(model_desc_, i);
    TensorDesc tensor_desc = TensorDesc(data_type, output_dim, data_format);
    output_tensor->push_back(tensor_desc);
  }
  return true;
}

bool AclModelClient::GetTensorFromDataset(
    std::vector<std::shared_ptr<ge::Tensor>>* output_tensor) {
  size_t device_output_num = aclmdlGetDatasetNumBuffers(output_dataset_);
  size_t tensor_output_num = reinterpret_cast<size_t>(output_tensor->size());
  if (device_output_num != tensor_output_num) {
    LOG(ERROR)
        << "[HUAWEI_ASCEND_NPU] output number not equal, device number is "
        << device_output_num << "tensor number is " << tensor_output_num;
    return false;
  }
  for (size_t i = 0; i < device_output_num; i++) {
    aclDataBuffer* buffer_device = aclmdlGetDatasetBuffer(output_dataset_, i);
    void* device_data = aclGetDataBufferAddr(buffer_device);
    uint32_t device_size = aclGetDataBufferSize(buffer_device);

    void* tensor_data = nullptr;
    ACL_CALL(aclrtMallocHost(&tensor_data, device_size));
    ACL_CALL(aclrtMemcpy(tensor_data,
                         device_size,
                         device_data,
                         device_size,
                         ACL_MEMCPY_DEVICE_TO_HOST));
    ATC_CALL(output_tensor->at(i)->SetData(
        reinterpret_cast<uint8_t*>(tensor_data), device_size));
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Get output tensor from dataset succeed.";
  return true;
}

void AclModelClient::CreateInputDataset(
    std::vector<std::shared_ptr<ge::Tensor>>* input_tensor) {
  input_dataset_ = aclmdlCreateDataset();
  if (input_dataset_ == nullptr) {
    LOG(ERROR) << "[HUAWEI_ASCEND_NPU] create input dataset failed!";
    return;
  }

  for (size_t i = 0; i < input_tensor->size(); i++) {
    auto item = input_tensor->at(i);
    size_t buffer_size = item->GetSize();
    void* buffer_device = nullptr;

    ACL_CALL(
        aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY));

    void* buffer_data = reinterpret_cast<void*>(item->GetData());
    auto ret = aclrtMemcpy(buffer_device,
                           buffer_size,
                           buffer_data,
                           buffer_size,
                           ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] input memcpy failed, buffer size is "
                 << buffer_size;
      ACL_CALL(aclrtFree(buffer_device));
      return;
    }
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(buffer_device, buffer_size);
    if (data_buffer == nullptr) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] output aclCreateDataBuffer failed!";
      ACL_CALL(aclrtFree(buffer_device));
      return;
    }
    if (aclmdlAddDatasetBuffer(input_dataset_, data_buffer) != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] input aclmdlAddDatasetBuffer failed!";
      ACL_CALL(aclrtFree(buffer_device));
      ACL_CALL(aclDestroyDataBuffer(data_buffer));
      return;
    }
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] CreateInputDataset succeed.";
}
void AclModelClient::CreateOutputDataset(
    std::vector<std::shared_ptr<ge::Tensor>>* output_tensor) {
  output_dataset_ = aclmdlCreateDataset();
  if (output_dataset_ == nullptr) {
    LOG(ERROR) << "[HUAWEI_ASCEND_NPU] create output dataset failed!";
    return;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  CHECK_EQ(output_size, output_tensor->size());
  for (size_t i = 0; i < output_size; i++) {
    size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    void* buffer_device = nullptr;
    ACL_CALL(
        aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(buffer_device, buffer_size);
    if (data_buffer == nullptr) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] output aclCreateDataBuffer failed!";
      ACL_CALL(aclrtFree(buffer_device));
      return;
    }
    if (aclmdlAddDatasetBuffer(output_dataset_, data_buffer) !=
        ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] output aclmdlAddDatasetBuffer failed!";
      ACL_CALL(aclrtFree(buffer_device));
      ACL_CALL(aclDestroyDataBuffer(data_buffer));
      return;
    }
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] CreateOutputDataset succeed.";
}

bool AclModelClient::ModelExecute(
    std::vector<std::shared_ptr<ge::Tensor>>* input_tensor,
    std::vector<std::shared_ptr<ge::Tensor>>* output_tensor) {
  // check model exists
  if (model_desc_ == nullptr) {
    LOG(ERROR)
        << "[HUAWEI_ASCEND_NPU] no model description, model execution failed!";
    return false;
  }
  // create input/output dataset
  CreateInputDataset(input_tensor);
  CreateOutputDataset(output_tensor);

  // model execution
  ACL_CALL(aclmdlExecute(model_id_, input_dataset_, output_dataset_));

  // get output
  if (!GetTensorFromDataset(output_tensor)) {
    LOG(ERROR) << "[HUAWEI_ASCEND_NPU] GetTensorFromDataset failed, modelId:"
               << model_id_;
    return false;
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] GetTensorFromDataset succeed, modelId:"
          << model_id_;

  return true;
}

void AclModelClient::DestroyDataset(aclmdlDataset** dataset) {
  if (*dataset == nullptr) {
    LOG(WARNING)
        << "[HUAWEI_ASCEND_NPU] no dataset exists, no need to destroy!";
    return;
  }

  size_t dataset_num = aclmdlGetDatasetNumBuffers(*dataset);
  for (size_t i = 0; i < dataset_num; i++) {
    aclDataBuffer* buffer_device = aclmdlGetDatasetBuffer(*dataset, i);
    void* device_data = aclGetDataBufferAddr(buffer_device);
    if (device_data == nullptr) {
      LOG(WARNING) << "[HUAWEI_ASCEND_NPU] failed to get data buffer!";
    } else {
      ACL_CALL(aclrtFree(device_data));
    }
    ACL_CALL(aclDestroyDataBuffer(buffer_device));
  }
  ACL_CALL(aclmdlDestroyDataset(*dataset));
  *dataset = nullptr;
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Destroy dataset success.";
}

bool AclModelClient::UnloadModel() {
  if (!load_flag_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] no need to unload model, load flag is "
                 << load_flag_;
    return true;
  }

  DestroyDataset(&input_dataset_);
  DestroyDataset(&output_dataset_);

  ACL_CALL(aclmdlUnload(model_id_));
  if (model_desc_ != nullptr) {
    ACL_CALL(aclmdlDestroyDesc(model_desc_));
    model_desc_ = nullptr;
  }

  if (model_memory_ptr_ != nullptr) {
    ACL_CALL(aclrtFree(model_memory_ptr_));
    model_memory_ptr_ = nullptr;
    model_memory_size_ = 0;
  }

  if (model_weight_ptr_ != nullptr) {
    ACL_CALL(aclrtFree(model_weight_ptr_));
    model_weight_ptr_ = nullptr;
    model_weight_size_ = 0;
  }
  load_flag_ = false;
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Unload model success, model id " << model_id_;
  return true;
}

uint32_t AclModelClient::num_devices() {
  uint32_t count = 0;
  ACL_CALL(aclrtGetDeviceCount(&count));
  return count;
}

}  // namespace huawei_ascend_npu
}  // namespace lite
}  // namespace paddle
