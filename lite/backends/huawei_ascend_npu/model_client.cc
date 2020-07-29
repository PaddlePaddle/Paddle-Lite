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

  auto ret = aclmdlQuerySizeFromMem(
      data, size, &model_memory_size_, &model_weight_size_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] query model size from memory failed!";
    return false;
  }
  ret = aclrtMalloc(
      &model_memory_ptr_, model_memory_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] malloc buffer for model memory "
                    "failed, require size is "
                 << model_memory_size_;
    return false;
  }
  ret = aclrtMalloc(
      &model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] malloc buffer for model weigth "
                    "failed, require size is "
                 << model_weight_size_;
    return false;
  }
  ret = aclmdlLoadFromMemWithMem(data,
                                 size,
                                 &model_id_,
                                 model_memory_ptr_,
                                 model_memory_size_,
                                 model_weight_ptr_,
                                 model_weight_size_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Load model from memory failed!";
    return false;
  }
  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] create model description failed!";
    return false;
  }
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] get model description failed!";
    return false;
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] AclModelClient LoadFromMem success.";
  load_flag_ = true;
  return true;
}

bool AclModelClient::LoadFromFile(const char* model_path) {
  if (load_flag_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] model is already loaded!";
    return true;
  }
  auto ret =
      aclmdlQuerySize(model_path, &model_memory_size_, &model_weight_size_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] query model size from file failed!";
    return false;
  }
  ret = aclrtMalloc(
      &model_memory_ptr_, model_memory_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] malloc buffer for model memory "
                    "failed, require size is "
                 << model_memory_size_;
    return false;
  }
  ret = aclrtMalloc(
      &model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] malloc buffer for model weigth "
                    "failed, require size is "
                 << model_weight_size_;
    return false;
  }
  ret = aclmdlLoadFromFileWithMem(model_path,
                                  &model_id_,
                                  model_memory_ptr_,
                                  model_memory_size_,
                                  model_weight_ptr_,
                                  model_weight_size_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Load model from file failed!";
    return false;
  }
  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] create model description failed!";
    return false;
  }
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] get model description failed!";
    return false;
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Loading model file success:" << model_path;
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
  VLOG(3) << "[HUAWEI_ASCEND_NPU] input numher is " << input_num;
  for (size_t i = 0; i < input_num; i++) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] printing input [" << i << "] ....";
    aclmdlIODims input_dim;
    aclmdlGetInputDims(model_desc_, i, &input_dim);
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    VLOG(3) << "[HUAWEI_ASCEND_NPU] data_type of inputs[" << i << "] is "
            << data_type;
    aclFormat data_format = aclmdlGetInputFormat(model_desc_, i);
    VLOG(3) << "[HUAWEI_ASCEND_NPU] data_format of inputs[" << i << "] is "
            << data_format;
    TensorDesc tensor_desc = TensorDesc(data_type, input_dim, data_format);
    input_tensor->push_back(tensor_desc);
  }

  size_t output_num = aclmdlGetNumOutputs(model_desc_);
  VLOG(3) << "[HUAWEI_ASCEND_NPU] output numher is " << output_num;
  for (size_t i = 0; i < output_num; i++) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] printing output [" << i << "] ....";
    aclmdlIODims output_dim;
    aclmdlGetOutputDims(model_desc_, i, &output_dim);
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    VLOG(3) << "[HUAWEI_ASCEND_NPU] data_type of outputs[" << i << "] is "
            << data_type;
    aclFormat data_format = aclmdlGetOutputFormat(model_desc_, i);
    VLOG(3) << "[HUAWEI_ASCEND_NPU] data_format of outputs[" << i << "] is "
            << data_format;
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
    aclError ret = aclrtMallocHost(&tensor_data, device_size);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] aclrtMallocHost failed, ret " << ret;
      return false;
    }
    ret = aclrtMemcpy(tensor_data,
                      device_size,
                      device_data,
                      device_size,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] aclrtMemcpy failed, ret " << ret;
      return false;
    }
    if (output_tensor->at(i)->SetData(reinterpret_cast<uint8_t*>(tensor_data),
                                      device_size) != ge::GRAPH_SUCCESS) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] SetData to output tensor failed";
      return false;
    }
  }
  VLOG(3)
      << "[HUAWEI_ASCEND_NPU] Get output tensor from output dataset succeed.";
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
    aclError ret =
        aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR)
          << "[HUAWEI_ASCEND_NPU] input malloc device buffer failed. size is "
          << buffer_size;
      return;
    }
    void* buffer_data = reinterpret_cast<void*>(item->GetData());
    ret = aclrtMemcpy(buffer_device,
                      buffer_size,
                      buffer_data,
                      buffer_size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] input memcpy failed, buffer size is "
                 << buffer_size;
      aclrtFree(buffer_device);
      return;
    }
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(buffer_device, buffer_size);
    if (data_buffer == nullptr) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] output aclCreateDataBuffer failed!";
      aclrtFree(buffer_device);
      return;
    }
    if (aclmdlAddDatasetBuffer(input_dataset_, data_buffer) != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] input aclmdlAddDatasetBuffer failed!";
      aclrtFree(buffer_device);
      aclDestroyDataBuffer(data_buffer);
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
    aclError ret =
        aclrtMalloc(&buffer_device, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR)
          << "[HUAWEI_ASCEND_NPU] output malloc device buffer failed. size is "
          << buffer_size;
      return;
    }
    aclDataBuffer* data_buffer =
        aclCreateDataBuffer(buffer_device, buffer_size);
    if (data_buffer == nullptr) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] output aclCreateDataBuffer failed!";
      aclrtFree(buffer_device);
      return;
    }
    if (aclmdlAddDatasetBuffer(output_dataset_, data_buffer) !=
        ACL_ERROR_NONE) {
      LOG(ERROR) << "[HUAWEI_ASCEND_NPU] output aclmdlAddDatasetBuffer failed!";
      aclrtFree(buffer_device);
      aclDestroyDataBuffer(data_buffer);
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
      LOG(WARNING)
          << "[HUAWEI_ASCEND_NPU] failed to get data buffer of deivce data!";
    } else {
      if (aclrtFree(device_data) != ACL_ERROR_NONE) {
        LOG(WARNING) << "[HUAWEI_ASCEND_NPU] failed to free deivce data!";
      }
    }
    if (aclDestroyDataBuffer(buffer_device) != ACL_ERROR_NONE) {
      LOG(WARNING)
          << "[HUAWEI_ASCEND_NPU] failed to destroy deivce data buffer!";
    }
  }
  if (aclmdlDestroyDataset(*dataset) != ACL_ERROR_NONE) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] failed to destroy dataset!";
  }
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

  aclError ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "unload model failed, model id is " << model_id_;
    return false;
  }
  if (model_desc_ != nullptr) {
    (void)aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }

  if (model_memory_ptr_ != nullptr) {
    aclrtFree(model_memory_ptr_);
    model_memory_ptr_ = nullptr;
    model_memory_size_ = 0;
  }

  if (model_weight_ptr_ != nullptr) {
    aclrtFree(model_weight_ptr_);
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
