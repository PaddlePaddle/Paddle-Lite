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

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "lite/kernels/mlu/bridges/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

// The Context of the converters which used for converting the ops of subgraph
// to the MLU IR graph
class Graph {
 public:
  Graph() { CNML_CALL(cnmlCreateFusionOp(&fusion_op_)); }

  ~Graph() {
    FreeConstData();
    CNML_CALL(cnmlDestroyFusionOp(&fusion_op_));
    for (auto op : ops_) {
      CNML_CALL(cnmlDestroyBaseOp(&op));
    }
  }

  // Data node
  std::shared_ptr<MLUTensor> AddNode(
      const std::string& name,
      std::vector<int64_t> shape,
      cnmlTensorType_t tensor_type = CNML_TENSOR,
      cnmlDataOrder_t data_order = CNML_NCHW,
      cnmlDataType_t mlu_dtype = CNML_DATA_FLOAT32,
      void* raw_ptr = nullptr);

  std::shared_ptr<MLUTensor> GetNode(const std::string& name) {
    CHECK(HasNode(name)) << "[MLU] Node " << name << " not found.";
    return nodes_.at(name);
  }

  bool HasNode(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

  void AddInput(std::shared_ptr<MLUTensor> tensor) {
    inputs_.push_back(tensor->mlu_tensor());
    input_tensors_.push_back(tensor);
  }

  void AddOutput(std::shared_ptr<MLUTensor> tensor) {
    outputs_.push_back(tensor->mlu_tensor());
    output_tensors_.push_back(tensor);
  }

  void FuseOp(cnmlBaseOp_t op) { CNML_CALL(cnmlFuseOp(op, fusion_op_)); }

  void Compile(cnmlCoreVersion_t core_version, int core_number) {
    CNML_CALL(cnmlSetFusionIO(fusion_op_,
                              inputs_.data(),
                              inputs_.size(),
                              outputs_.data(),
                              outputs_.size()));
    CNML_CALL(cnmlSetFusionOpCorenum(fusion_op_, core_number));
    CNML_CALL(cnmlSetFusionOpCoreVersion(fusion_op_, core_version));
    CNML_CALL(cnmlCompileFusionOp_V2(fusion_op_));
    for (auto in : input_tensors_) {
      input_addrs_.push_back(in->mlu_data());
    }
    for (auto out : output_tensors_) {
      output_addrs_.push_back(out->mlu_data());
    }
  }

  void Compute(cnrtInvokeFuncParam_t forward_param, cnrtQueue_t que) {
    CNML_CALL(cnmlComputeFusionOpForward_V3(fusion_op_,
                                            input_addrs_.data(),
                                            input_addrs_.size(),
                                            output_addrs_.data(),
                                            output_addrs_.size(),
                                            &forward_param,
                                            que));
    CNRT_CALL(cnrtSyncQueue(que));
  }

  template <typename T>
  void* RegisterConstData(size_t len) {
    void* addr = malloc(len * sizeof(T));
    const_data_storage_.push_back(addr);
    return addr;
  }

  void FreeConstData() {
    for (auto& addr : const_data_storage_) {
      free(addr);
    }
  }

  void BindConstRawData(std::string tensor_name,
                        const float* data,
                        size_t len,
                        bool alloc = true) {
    void* alloc_data;
    if (fp_type_ == CNML_DATA_FLOAT32) {
      if (alloc) {
        alloc_data = RegisterConstData<float>(len);
        memcpy(alloc_data, data, len * sizeof(float));
      } else {
        alloc_data = const_cast<void*>(static_cast<const void*>(data));
      }
      CNML_CALL(cnmlBindConstData_V2(
          nodes_[tensor_name]->mlu_tensor(), alloc_data, false));
    } else if (fp_type_ == CNML_DATA_FLOAT16) {
      void* data_fp16 = RegisterConstData<::paddle::lite::fluid::float16>(len);
      CNRT_CALL(
          cnrtCastDataType(const_cast<void*>(static_cast<const void*>(data)),
                           CNRT_FLOAT32,
                           data_fp16,
                           CNRT_FLOAT16,
                           len,
                           nullptr));
      CNML_CALL(cnmlBindConstData_V2(
          nodes_[tensor_name]->mlu_tensor(), data_fp16, false));
    } else {
      CHECK(0);
    }
  }

  void BindConstData(std::string tensor_name, ::paddle::lite::Tensor* tensor) {
    const float* data = tensor->data<float>();
    size_t len = tensor->data_size();
    if (fp_type_ == CNML_DATA_FLOAT32) {
      CNML_CALL(cnmlBindConstData_V2(
          nodes_[tensor_name]->mlu_tensor(),
          const_cast<void*>(static_cast<const void*>(data)),
          false));
    } else if (fp_type_ == CNML_DATA_FLOAT16) {
      auto* data_fp16 = tensor->mutable_data<::paddle::lite::fluid::float16>();
      for (size_t i = 0; i < len; ++i) {
        data_fp16[i] = static_cast<::paddle::lite::fluid::float16>(data[i]);
      }
      CNML_CALL(cnmlBindConstData_V2(nodes_[tensor_name]->mlu_tensor(),
                                     static_cast<void*>(data_fp16),
                                     false));
    } else {
      CHECK(0);
    }
  }

  void SetComputingDataType(cnmlBaseOp_t op,
                            cnmlTensor_t tensor,
                            float scale,
                            cnmlDataType_t data_type = CNML_DATA_INT8) {
    cnmlQuantizedParam_t quant_param;
    CNML_CALL(
        cnmlCreateQuantizedParam(&quant_param, scale2position(scale), 1, 0.0));
    CNML_CALL(
        cnmlSetOperationComputingDataType(op, tensor, data_type, quant_param));
    CNML_CALL(cnmlDestroyQuantizedParam(&quant_param));
  }

  void SetFPType(::paddle::lite_api::PrecisionType type) {
    switch (type) {
      case ::paddle::lite_api::PrecisionType::kFP16:
        fp_type_ = CNML_DATA_FLOAT16;
        break;
      case ::paddle::lite_api::PrecisionType::kFloat:
        fp_type_ = CNML_DATA_FLOAT32;
        break;
      default:
        CHECK(0);
    }
  }

  cnmlDataType_t FPType() { return fp_type_; }

 private:
  cnmlDataType_t fp_type_{CNML_DATA_FLOAT32};
  std::unordered_map<std::string, std::shared_ptr<MLUTensor>> nodes_;
  std::vector<cnmlTensor_t> inputs_;
  std::vector<cnmlTensor_t> outputs_;
  std::vector<void*> input_addrs_;
  std::vector<void*> output_addrs_;
  std::vector<std::shared_ptr<MLUTensor>> input_tensors_;
  std::vector<std::shared_ptr<MLUTensor>> output_tensors_;
  std::vector<cnmlBaseOp_t> ops_;
  cnmlFusionOp_t fusion_op_;
  std::vector<void*> const_data_storage_;
};

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
