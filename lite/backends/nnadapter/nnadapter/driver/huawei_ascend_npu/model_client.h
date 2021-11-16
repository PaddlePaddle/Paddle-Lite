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

#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "acl/acl.h"
#include "acl/acl_prof.h"
#include "core/hal/types.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace nnadapter {
namespace huawei_ascend_npu {

typedef enum {
  ASCEND_NPU_CONST_SHAPE = -1,
  ASCEND_NPU_DYNAMIC_BATCH = 0,
  ASCEND_NPU_DYNAMIC_HEIGHT_WEIGHT = 1,
  ASCEND_NPU_DYNAMIC_N_DIM = 2,
} AscendNPUDynamicShapeMode;

class AclModelClient {
 public:
  explicit AclModelClient(int device_id,
                          const std::string& profiling_file_path);
  ~AclModelClient();

  bool LoadModel(const void* data, uint32_t size);
  void UnloadModel();
  bool GetModelIOTensorDim(std::vector<ge::TensorDesc>* input_tensor_descs,
                           std::vector<ge::TensorDesc>* output_tensor_descs);
  bool Process(uint32_t input_count,
               std::vector<NNAdapterOperandType>* input_types,
               hal::Argument* input_arguments,
               uint32_t output_count,
               std::vector<NNAdapterOperandType>* output_types,
               hal::Argument* output_arguments,
               AscendNPUDynamicShapeMode dynamic_shape_mode);

 private:
  void InitAclClientEnv(int device_id);
  void FinalizeAclClientEnv();
  void InitAclProfilingEnv(const std::string& profiling_file_path);
  void FinalizeAclProfilingEnv();
  bool CreateModelIODataset();
  void DestroyDataset(aclmdlDataset** dataset);
  void ProfilingStart();
  void ProfilingEnd();

 private:
  int device_id_{0};
  aclrtContext context_{nullptr};
  uint32_t model_id_{0};
  size_t model_memory_size_;
  size_t model_weight_size_;
  void* model_memory_ptr_;
  void* model_weight_ptr_;
  aclmdlDesc* model_desc_{nullptr};
  aclmdlDataset* input_dataset_{nullptr};
  aclmdlDataset* output_dataset_{nullptr};
  aclprofConfig* config_{nullptr};
};

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
