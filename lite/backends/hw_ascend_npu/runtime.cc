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

#include "lite/backends/hw_ascend_npu/runtime.h"
#include "lite/backends/hw_ascend_npu/target_wrapper.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace hw_ascend_npu {
HWAscendNPURuntime::HWAscendNPURuntime(
    std::shared_ptr<uint8_t> model_buff_built, size_t model_buff_size) {
  model_loaded_ = (0 == LoadModelFromMem(model_buff_built, model_buff_size));
}

HWAscendNPURuntime::~HWAscendNPURuntime() {
  UnloadModel();
  DestroyDesc();
  DestroyInput();
  DestroyOutput();
}

int HWAscendNPURuntime::LoadModelFromMem(
    std::shared_ptr<uint8_t> model_buff_built, size_t model_buff_size) {
  if (model_loaded_) {
    LOG(ERROR) << "[HWAscendNPU]: Has already loaded a model";
    return 0;
  }
  aclError ret = aclmdlQuerySizeFromMem(model_buff_built.get(),
                                        model_buff_size,
                                        &model_size_,
                                        &model_weights_size_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Can't query size from a built model buffer, "
                  "error code: "
               << ret << ", model buffer size: " << model_buff_size;
    return ret;
  }

  LOG(INFO) << "[HWAscendNPU]: Query model info success, model_size: "
            << model_size_ << ", model weights_size_: " << model_weights_size_;

  ret = aclrtMalloc(&model_ptr_, model_size_, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Can not allocate a device memory for model, "
                  "error code: "
               << ret;
    return ret;
  }

  ret = aclrtMalloc(
      &model_weights_ptr_, model_weights_size_, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Can not allocate a device memory for model "
                  "weights, error code: "
               << ret;
    return ret;
  }

  ret = aclmdlLoadFromMemWithMem(model_buff_built.get(),
                                 model_buff_size,
                                 &model_id_,
                                 model_ptr_,
                                 model_size_,
                                 model_weights_ptr_,
                                 model_weights_size_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Can not Load model from memory, error code: "
               << ret;
    return ret;
  }

  model_desc_ = aclmdlCreateDesc();
  if (model_desc_ == nullptr) {
    LOG(ERROR) << "HWAscendNPU]: Can not create model descriptor.";
    return ACL_ERROR_FAILURE;
  }

  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Can not get model descriptor from model, "
                  "error code: "
               << ret;
    return ret;
  }
  return ret;
}

int HWAscendNPURuntime::CreateInput(const std::vector<DDim>& idims) {
  if (itensors_ != nullptr) {
    DestroyInput();
  }
  itensors_ = aclmdlCreateDataset();
  if (itensors_ == nullptr) {
    LOG(ERROR) << "[HWAscendNPU]: Can not create input dataset";
    return ACL_ERROR_FAILURE;
  }

  for (auto& dim : idims) {
    void* buff_dev_ptr = nullptr;
    CHECK(ACL_ERROR_NONE == aclrtMalloc(&buff_dev_ptr,
                                        dim.production(),
                                        ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* input_data_buffer =
        aclCreateDataBuffer(buff_dev_ptr, dim.production());
    CHECK(input_data_buffer != nullptr);
    CHECK(ACL_ERROR_NONE ==
          aclmdlAddDatasetBuffer(itensors_, input_data_buffer));
  }
  return 0;
}

int HWAscendNPURuntime::CreateOutput(const std::vector<DDim>& odims) {
  if (otensors_ != nullptr) {
    DestroyOutput();
  }
  otensors_ = aclmdlCreateDataset();
  if (otensors_ == nullptr) {
    LOG(ERROR) << "[HWAscendNPU]: Can not create output dataset";
    return ACL_ERROR_FAILURE;
  }

  for (auto& dim : odims) {
    void* buff_dev_ptr = nullptr;
    CHECK(ACL_ERROR_NONE == aclrtMalloc(&buff_dev_ptr,
                                        dim.production(),
                                        ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* output_data_buffer =
        aclCreateDataBuffer(buff_dev_ptr, dim.production());
    CHECK(output_data_buffer != nullptr);
    CHECK(ACL_ERROR_NONE ==
          aclmdlAddDatasetBuffer(otensors_, output_data_buffer));
  }
  return 0;
}

void HWAscendNPURuntime::UnloadModel() {
  if (!model_loaded_) {
    LOG(ERROR) << "[HWAscendNPU]: No model has been loaded";
    return;
  }
  aclError ret = ACL_ERROR_NONE;
  ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Unload a model, error code: " << ret;
    return;
  }

  if (model_ptr_) {
    aclrtFree(model_ptr_);
    model_ptr_ = nullptr;
  }

  if (model_weights_ptr_) {
    aclrtFree(model_weights_ptr_);
    model_weights_ptr_ = nullptr;
  }
  model_loaded_ = false;
}

void HWAscendNPURuntime::DestroyDesc() {
  if (model_desc_) {
    (void)aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
}

void HWAscendNPURuntime::DestroyInput() {
  if (itensors_ == nullptr) {
    return;
  }
  size_t buf_num = aclmdlGetDatasetNumBuffers(itensors_);
  for (size_t i = 0; i < buf_num; ++i) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(itensors_, i);
    aclDestroyDataBuffer(data_buffer);
  }
  aclmdlDestroyDataset(itensors_);
  itensors_ = nullptr;
}

void HWAscendNPURuntime::DestroyOutput() {
  if (otensors_ == nullptr) {
    return;
  }
  size_t buf_num = aclmdlGetDatasetNumBuffers(otensors_);
  for (size_t i = 0; i < buf_num; ++i) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(otensors_, i);
    aclDestroyDataBuffer(data_buffer);
  }
  aclmdlDestroyDataset(otensors_);
  otensors_ = nullptr;
}

int HWAscendNPURuntime::SetInput(const std::vector<Tensor*>& itensors,
                                 const std::vector<DDim>& idims) {
  CHECK(itensors.size() == idims.size());
  size_t input_tensor_num = itensors.size();
  for (size_t i = 0; i < input_tensor_num; ++i) {
    CHECK(itensors[i]->memory_size() == idims[i].production());
  }
  size_t num_buffers_in_dataset = aclmdlGetDatasetNumBuffers(itensors_);
  if (num_buffers_in_dataset != input_tensor_num) {
    if (0 != CreateInput(idims)) {
      return -1;
    }
  } else {
    bool need_to_create_input = false;
    for (size_t i = 0; i < num_buffers_in_dataset; ++i) {
      aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(itensors_, i);
      int64_t buf_size = aclGetDataBufferSize(data_buffer);
      if (buf_size != idims[i].production()) {
        need_to_create_input = true;
      }
    }
    if (need_to_create_input && 0 != CreateInput(idims)) {
      return -1;
    }
  }

  // copy input data from host to device
  for (size_t i = 0; i < input_tensor_num; ++i) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(itensors_, i);
    void* buf_dev_ptr = aclGetDataBufferAddr(data_buffer);
    TargetWrapperHWAscendNPU::MemcpySync(buf_dev_ptr,
                                         itensors[i]->raw_data(),
                                         itensors[i]->memory_size(),
                                         IoDirection::HtoD);
  }
  return 0;
}

void HWAscendNPURuntime::GetOutput(const std::vector<Tensor*>* otensors_ptr) {
  CHECK(otensors_ptr != nullptr);
  size_t num_output = aclmdlGetDatasetNumBuffers(otensors_);
  const std::vector<Tensor*> otensors = *otensors_ptr;

  CHECK(num_output == otensors.size());
  for (size_t i = 0; i < num_output; ++i) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(otensors_, i);
    TargetWrapperHWAscendNPU::MemcpySync(otensors[i]->raw_data(),
                                         aclGetDataBufferAddr(data_buffer),
                                         aclGetDataBufferSize(data_buffer),
                                         IoDirection::DtoH);
  }
}

int HWAscendNPURuntime::Process() {
  aclError ret = aclmdlExecute(model_id_, itensors_, otensors_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU]: Execute model failed, model_id: " << model_id_
               << ", error code: " << ret;
  }
  return ret;
}

int HWAscendNPURuntime::GetModelIOTensorDim(std::vector<TensorDesc>* idims,
                                            std::vector<TensorDesc>* odims) {
  aclError ret = ACL_ERROR_NONE;
  size_t num_inputs = aclmdlGetNumInputs(model_desc_);
  size_t num_outputs = aclmdlGetNumOutputs(model_desc_);
  for (size_t i = 0; i < num_inputs; ++i) {
    aclmdlIODims dims;
    if (ret != aclmdlGetInputDims(model_desc_, i, &dims)) {
      LOG(ERROR) << "[HWAscendNPU]: Get input dims failed, index: " << i;
      return ret;
    }
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    aclFormat format = aclmdlGetInputFormat(model_desc_, i);

    idims->push_back(TensorDesc(data_type, dims, format));
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    aclmdlIODims dims;
    if (ret != aclmdlGetOutputDims(model_desc_, i, &dims)) {
      LOG(ERROR) << "[HWAscendNPU]: Get output dims failed, index: " << i;
      return ret;
    }
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    aclFormat format = aclmdlGetOutputFormat(model_desc_, i);

    odims->push_back(TensorDesc(data_type, dims, format));
  }
  return 0;
}
}  // namespace hw_ascend_npu
}  // namespace lite
}  // namespace paddle
