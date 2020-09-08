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
#include "lite/utils/env.h"
#include "lite/utils/macros.h"

#define PRINT_HW_TIME false

#if PRINT_HW_TIME
#include <mutex>  //NOLINT
#endif

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

// The Context of the converters which used for converting the ops of subgraph
// to the MLU IR graph
class Graph {
 public:
  Graph() {
    CNML_CALL(cnmlCreateFusionOp(&fusion_op_));
#if PRINT_HW_TIME
    CNRT_CALL(cnrtCreateNotifier(&notifier_start_));
    CNRT_CALL(cnrtCreateNotifier(&notifier_end_));
#endif
  }
  ~Graph() {
    FreeConstData();
    CNML_CALL(cnmlDestroyFusionOp(&fusion_op_));
#if PRINT_HW_TIME
    CNRT_CALL(cnrtDestroyNotifier(&notifier_start_));
    CNRT_CALL(cnrtDestroyNotifier(&notifier_end_));
    double total_time = 0;
    if (!time_log_.empty()) {
      for (auto& f : time_log_) {
        total_time += f;
      }
      std::cout << "cnml hardware time for " << time_log_.size()
                << " process:" << total_time / time_log_.size() << std::endl;
    }
#endif
  }
  // Data node
  std::shared_ptr<MLUTensor> AddNode(
      const std::string& name,
      std::vector<int64_t> shape,
      cnmlTensorType_t tensor_type = CNML_TENSOR,
      cnmlDataOrder_t shape_order = CNML_NCHW,
      cnmlDataType_t mlu_dtype = CNML_DATA_FLOAT32,
      cnmlDataOrder_t data_order = CNML_NHWC,
      void* raw_ptr = nullptr);

  std::shared_ptr<MLUTensor> GetNode(const std::string& name) {
    CHECK(HasNode(name)) << "[MLU] Node " << name << " not found.";
    return nodes_.at(name);
  }

  bool HasNode(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

  void AddInput(std::shared_ptr<MLUTensor> tensor,
                bool disable_batch_size_changeable = true) {
    inputs_.push_back(tensor->mlu_tensor());
    input_tensors_.push_back(tensor);
    if (!disable_batch_size_changeable) {
      constexpr int input_dimNb = 4;
      bool input_dim_mutable[4] = {true, false, false, false};
      CNML_CALL(cnmlSetTensorDimMutable(
          tensor->mlu_tensor(), input_dim_mutable, input_dimNb));
    }
  }

  void AddOutput(std::shared_ptr<MLUTensor> tensor) {
    outputs_.push_back(tensor->mlu_tensor());
    output_tensors_.push_back(tensor);
  }

  std::vector<std::shared_ptr<MLUTensor>>* MutableInputs() {
    return &input_tensors_;
  }

  std::vector<std::shared_ptr<MLUTensor>>* MutableOutputs() {
    return &output_tensors_;
  }
  void GenOfflineModel(const std::string& name) {
    cnmlModel_t model;
    const std::string& symbol = "subnet0";
    const auto& filename = name + ".offline.cambricon";
    CNML_CALL(cnmlCreateModel(&model, filename.c_str()));
    CNML_CALL(cnmlAddFusionOpToModel(model, fusion_op_, symbol.c_str()));
    CNML_CALL(cnmlSaveModel(model, filename.c_str()));
    CNML_CALL(cnmlDestroyModel(model));
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
  }

#define MEASURE_HWTIME_START(que)                       \
  do {                                                  \
    CNRT_CALL(cnrtPlaceNotifier(notifier_start_, que)); \
  } while (0)

#define MEASURE_HWTIME_END(que)                                                \
  do {                                                                         \
    static LITE_THREAD_LOCAL float hw_time;                                    \
    CNRT_CALL(cnrtPlaceNotifier(notifier_end_, que));                          \
    CNRT_CALL(cnrtSyncQueue(que));                                             \
    CNRT_CALL(cnrtNotifierDuration(notifier_start_, notifier_end_, &hw_time)); \
    hw_time /= 1000.0f;                                                        \
    DLOG(INFO) << "cnml hardware time " << hw_time << "ms" << std::endl;       \
    std::lock_guard<std::mutex> lk(time_mut_);                                 \
    time_log_.push_back(hw_time);                                              \
  } while (0)

  void Compute(cnrtInvokeFuncParam_t forward_param, cnrtQueue_t que) {
    input_addrs_.resize(input_tensors_.size());
    output_addrs_.resize(output_tensors_.size());
    for (size_t i = 0; i < input_addrs_.size(); ++i) {
      input_addrs_[i] = input_tensors_[i]->mlu_data();
    }
    for (size_t i = 0; i < output_addrs_.size(); ++i) {
      output_addrs_[i] = output_tensors_[i]->mlu_data();
    }

#if PRINT_HW_TIME
    MEASURE_HWTIME_START(que);
#endif
    CNML_CALL(cnmlComputeFusionOpForward_V3(fusion_op_,
                                            input_addrs_.data(),
                                            input_addrs_.size(),
                                            output_addrs_.data(),
                                            output_addrs_.size(),
                                            &forward_param,
                                            que));
#if PRINT_HW_TIME
    MEASURE_HWTIME_END(que);
#endif
  }

  void Compute(cnrtQueue_t que,
               const std::vector<std::shared_ptr<MLUTensor>>& in,
               const std::vector<std::shared_ptr<MLUTensor>>& out) {
    std::vector<cnmlTensor_t> in_tensor;
    std::vector<cnmlTensor_t> out_tensor;
    input_addrs_.resize(in.size());
    output_addrs_.resize(out.size());
    for (size_t i = 0; i < input_addrs_.size(); ++i) {
      input_addrs_[i] = in[i]->mlu_data();
      in_tensor.push_back(in[i]->mlu_tensor());
    }
    for (size_t i = 0; i < output_addrs_.size(); ++i) {
      output_addrs_[i] = out[i]->mlu_data();
      out_tensor.push_back(out[i]->mlu_tensor());
    }

#if PRINT_HW_TIME
    MEASURE_HWTIME_START(que);
#endif
    /* Because of using cnmlSetTensorDimMutable, cnmlComputeFusionOpForward_V3
     * -> cnmlComputeFusionOpForward_V4 */
    CNML_CALL(cnmlComputeFusionOpForward_V4(fusion_op_,
                                            &in_tensor[0],
                                            input_addrs_.data(),
                                            input_addrs_.size(),
                                            &out_tensor[0],
                                            output_addrs_.data(),
                                            output_addrs_.size(),
                                            que,
                                            NULL));
#if PRINT_HW_TIME
    MEASURE_HWTIME_END(que);
#endif
  }
#undef MEASURE_HWTIME_START
#undef MEASURE_HWTIME_END

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
      void* data_fp16 = RegisterConstData<paddle::lite::fluid::float16>(len);
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

  void BindConstData(std::string tensor_name, paddle::lite::Tensor* tensor) {
    const float* data = tensor->data<float>();
    size_t len = tensor->data_size();
    if (fp_type_ == CNML_DATA_FLOAT32) {
      CNML_CALL(cnmlBindConstData_V2(
          nodes_[tensor_name]->mlu_tensor(),
          const_cast<void*>(static_cast<const void*>(data)),
          false));
    } else if (fp_type_ == CNML_DATA_FLOAT16) {
      void* data_fp16 = RegisterConstData<paddle::lite::fluid::float16>(len);
      CNRT_CALL(
          cnrtCastDataType(const_cast<void*>(static_cast<const void*>(data)),
                           CNRT_FLOAT32,
                           data_fp16,
                           CNRT_FLOAT16,
                           len,
                           nullptr));
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
    int pos = scale2position(scale);
    auto cnml_scale = pow(2, pos) * scale;
    VLOG(5) << "[cnml quantized param] pos: " << pos
            << "\tscale: " << cnml_scale << std::endl;
    CNML_CALL(cnmlCreateQuantizedParam(&quant_param, pos, cnml_scale, 0.0));
    CNML_CALL(
        cnmlSetOperationComputingDataType(op, tensor, data_type, quant_param));
    CNML_CALL(cnmlDestroyQuantizedParam(&quant_param));
  }

  void SetFPType(paddle::lite_api::PrecisionType type) {
    origin_fp_type_ = type;
    switch (type) {
      case paddle::lite_api::PrecisionType::kFP16:
        fp_type_ = CNML_DATA_FLOAT16;
        break;
      case paddle::lite_api::PrecisionType::kFloat:
        fp_type_ = CNML_DATA_FLOAT32;
        break;
      default:
        CHECK(0);
    }
  }

  cnmlDataType_t FPType() { return fp_type_; }

 private:
  cnmlDataType_t fp_type_{CNML_DATA_FLOAT32};
  paddle::lite_api::PrecisionType origin_fp_type_{PRECISION(kFloat)};
  std::unordered_map<std::string, std::shared_ptr<MLUTensor>> nodes_;
  std::vector<cnmlTensor_t> inputs_;
  std::vector<cnmlTensor_t> outputs_;
  std::vector<void*> input_addrs_;
  std::vector<void*> output_addrs_;
  std::vector<std::shared_ptr<MLUTensor>> input_tensors_;
  std::vector<std::shared_ptr<MLUTensor>> output_tensors_;
  cnmlFusionOp_t fusion_op_;
  std::vector<void*> const_data_storage_;
#if PRINT_HW_TIME
  cnrtNotifier_t notifier_start_{}, notifier_end_{};
  std::mutex time_mut_;
  std::vector<float> time_log_;
#endif
};

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
