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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lite/backends/huawei_ascend_npu/utils.h"

namespace paddle {
namespace lite {
namespace huawei_ascend_npu {

class TensorDesc {
 public:
  TensorDesc(const std::string name,
             aclDataType data_type,
             aclmdlIODims dims,
             aclFormat format) {
    if (format == ACL_FORMAT_NHWC) {
      dim_order[1] = 3;
      dim_order[2] = 1;
      dim_order[3] = 2;
    }
    // create ge::Tensordesc
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Getting tensor name : " << name;
    ge_tensor_desc_ = new ge::TensorDesc(
        GetGeShape(dims), GetGeFormat(format), GetGeDataType(data_type));
    ge_tensor_desc_->SetName(name);
    CHECK(ge_tensor_desc_ != nullptr);
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Getting data shape : " << repr();
  }
  ~TensorDesc() { ge_tensor_desc_ = nullptr; }

  const ge::TensorDesc& GetGeTensorDesc() const { return *ge_tensor_desc_; }

  std::string repr() const {
    STL::stringstream ss;
    size_t dim_size = ge_tensor_desc_->GetShape().GetDimNum();
    if (dim_size == 0) {
      ss << "{}";
      return ss.str();
    }
    ss << "{";
    for (size_t i = 0; i < dim_size - 1; i++) {
      ss << ge_tensor_desc_->GetShape().GetDim(i) << ",";
    }
    ss << ge_tensor_desc_->GetShape().GetDim(dim_size - 1);
    ss << "}";
    return ss.str();
  }

  int64_t production() const {
    return ge_tensor_desc_->GetShape().GetShapeSize();
  }

 private:
  ge::Shape GetGeShape(aclmdlIODims dims) {
    auto shape_data = std::vector<int64_t>({1L, 1L, 1L, 1L});
    shape_data.resize(dims.dimCount);
    ge::Shape ge_shape(shape_data);
    for (size_t i = 0; i < dims.dimCount; i++) {
      ATC_CALL(ge_shape.SetDim(i, dims.dims[i]));
    }
    return ge_shape;
  }
  ge::Format GetGeFormat(aclFormat format) {
    ge::Format ge_format = ge::FORMAT_NCHW;
    switch (format) {
      case ACL_FORMAT_NCHW:
        ge_format = ge::FORMAT_NCHW;
        break;
      case ACL_FORMAT_NHWC:
        ge_format = ge::FORMAT_NHWC;
        break;
      case ACL_FORMAT_ND:
        ge_format = ge::FORMAT_ND;
        break;
      default:
        LOG(FATAL) << "[HUAWEI_ASCEND_NPU] format not supported:" << format;
        break;
    }
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Getting data format : "
            << CvtFormat(ge_format);
    return ge_format;
  }
  ge::DataType GetGeDataType(aclDataType data_type) {
    ge::DataType ge_datatype = ge::DT_FLOAT;
    switch (data_type) {
      case ACL_FLOAT:
        ge_datatype = ge::DT_FLOAT;
        break;
      case ACL_FLOAT16:
        ge_datatype = ge::DT_FLOAT16;
        break;
      case ACL_INT8:
        ge_datatype = ge::DT_INT8;
        break;
      case ACL_INT16:
        ge_datatype = ge::DT_INT16;
        break;
      case ACL_INT32:
        ge_datatype = ge::DT_INT32;
        break;
      case ACL_INT64:
        ge_datatype = ge::DT_INT64;
        break;
      case ACL_BOOL:
        ge_datatype = ge::DT_BOOL;
        break;
      default:
        LOG(FATAL) << "[HUAWEI_ASCEND_NPU] data type not supported!";
        break;
    }
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Getting data type : "
            << CvtDataType(ge_datatype);
    return ge_datatype;
  }

 private:
  ge::TensorDesc* ge_tensor_desc_{nullptr};
  // n c h w order, default to ACL_FORMAT_NCHW
  std::vector<size_t> dim_order{0, 1, 2, 3};
};

class AclModelClient {
 public:
  explicit AclModelClient(int device_id) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Creating Huawei Ascend Device: "
            << device_id;
    device_num_ = num_devices();
    if (device_id < 0 || device_id >= device_num_) {
      LOG(FATAL) << "Failed with invalid device id " << device_id;
      return;
    }
    device_id_ = device_id;
    ACL_CALL(aclrtSetDevice(device_id_));
  }

  ~AclModelClient() {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Unloading model, model id is: "
            << model_id_;
    UnloadModel();
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Destroying Huawei Ascend Device: "
            << device_id_;
    ACL_CALL(aclrtResetDevice(device_id_));
  }

  bool LoadFromMem(const void* data, uint32_t size);
  bool LoadFromFile(const char* model_path);
  bool GetModelIOTensorDim(std::vector<TensorDesc>* input_tensor,
                           std::vector<TensorDesc>* output_tensor);
  bool ModelExecute(std::vector<std::shared_ptr<ge::Tensor>>* input_tensor,
                    std::vector<std::shared_ptr<ge::Tensor>>* output_tensor);

 private:
  void CreateInputDataset(
      std::vector<std::shared_ptr<ge::Tensor>>* input_tensor);
  void CreateOutputDataset(
      std::vector<std::shared_ptr<ge::Tensor>>* output_tensor);
  bool GetTensorFromDataset(
      std::vector<std::shared_ptr<ge::Tensor>>* output_tensor);
  void DestroyDataset(aclmdlDataset** dataset);
  void UnloadModel();

 private:
  uint32_t num_devices();

 private:
  int device_id_{0};
  int device_num_{0};
  aclrtContext context_{nullptr};
  bool load_flag_{false};
  uint32_t model_id_{0};
  size_t model_memory_size_;
  size_t model_weight_size_;
  void* model_memory_ptr_;
  void* model_weight_ptr_;
  aclmdlDesc* model_desc_{nullptr};
  aclmdlDataset* input_dataset_{nullptr};
  aclmdlDataset* output_dataset_{nullptr};
};

}  // namespace huawei_ascend_npu
}  // namespace lite
}  // namespace paddle
