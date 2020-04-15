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

#pragma once

// hw_ascend_npu runtime library
#include <acl/acl.h>
#include <acl/tensor.h>
#include <memory>
#include <vector>
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace hw_ascend_npu {
class TensorDesc {
 public:
  TensorDesc(aclDataType data_type, aclmdlIODims dims, aclFormat format) {
    tensor_desc_ =
        aclCreateTensorDesc(data_type, dims.dimCount, dims.dims, format);
    CHECK(tensor_desc_ != nullptr);
    aclSetTensorDescName(tensor_desc_, dims.name);
    if (format == ACL_FORMAT_NHWC) {
      dim_order[1] = 3;
      dim_order[2] = 1;
      dim_order[3] = 2;
    }
  }
  ~TensorDesc() {
    if (tensor_desc_ != nullptr) {
      aclDestroyTensorDesc(tensor_desc_);
      tensor_desc_ = nullptr;
    }
  }
  uint32_t GetNumber() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[0]));
  }
  uint32_t GetChannel() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[1]));
  }
  uint32_t GetHeight() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[2]));
  }
  uint32_t GetWidth() const {
    return static_cast<uint32_t>(
        aclGetTensorDescDim(tensor_desc_, dim_order[3]));
  }
  const aclTensorDesc& GetTensorDesc() const { return *tensor_desc_; }

 private:
  aclTensorDesc* tensor_desc_{nullptr};
  // n c h w order, default to ACL_FORMAT_NCHW
  std::vector<uint32_t> dim_order{0, 1, 2, 3};
};

class HWAscendNPURuntime {
 public:
  HWAscendNPURuntime(std::shared_ptr<uint8_t> model_buff_built,
                     size_t model_buff_size);
  ~HWAscendNPURuntime();

  int SetInput(const std::vector<Tensor*>& itensors,
               const std::vector<DDim>& idims);
  void GetOutput(const std::vector<Tensor*>* otensors_ptr);
  int Process();
  bool model_loaded() const { return model_loaded_; }
  int CreateInput(const std::vector<DDim>& idims);
  int CreateOutput(const std::vector<DDim>& odims);
  int GetModelIOTensorDim(std::vector<TensorDesc>* idims,
                          std::vector<TensorDesc>* odims);

 private:
  int LoadModelFromMem(std::shared_ptr<uint8_t> model_buff_built,
                       size_t model_buff_size);

  void UnloadModel();
  void DestroyDesc();
  void DestroyInput();
  void DestroyOutput();

 private:
  aclmdlDataset* itensors_{nullptr};
  aclmdlDataset* otensors_{nullptr};
  uint32_t model_id_{0};
  void* model_ptr_{nullptr};
  void* model_weights_ptr_{nullptr};
  size_t model_size_{0};
  size_t model_weights_size_{0};
  bool model_loaded_{false};
  aclmdlDesc* model_desc_;
};
}  // namespace hw_ascend_npu
}  // namespace lite
}  // namespace paddle
