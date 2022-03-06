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
#include "core/types.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace nnadapter {
namespace huawei_ascend_npu {

typedef enum {
  DYNAMIC_SHAPE_MODE_NONE = -1,
  DYNAMIC_SHAPE_MODE_BTACH_SIZE = 0,
  DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH = 1,
  DYNAMIC_SHAPE_MODE_N_DIMS = 2,
} DynamicShapeMode;

class AclModelClient {
 public:
  explicit AclModelClient(int device_id,
                          const std::string& profiling_file_path);
  ~AclModelClient();

  bool LoadModel(const void* data, size_t size);
  void UnloadModel();
  bool GetModelIOTensorDim(std::vector<ge::TensorDesc>* input_tensor_descs,
                           std::vector<ge::TensorDesc>* output_tensor_descs);
  bool Process(uint32_t input_count,
               std::vector<NNAdapterOperandType>* input_types,
               core::Argument* input_arguments,
               uint32_t output_count,
               std::vector<NNAdapterOperandType>* output_types,
               core::Argument* output_arguments,
               DynamicShapeMode dynamic_shape_mode);

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
  aclmdlDesc* model_desc_{nullptr};
  aclmdlDataset* input_dataset_{nullptr};
  aclmdlDataset* output_dataset_{nullptr};
  aclprofConfig* config_{nullptr};
};

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
