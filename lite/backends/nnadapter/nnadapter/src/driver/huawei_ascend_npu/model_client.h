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
  DYNAMIC_SHAPE_MODE_BATCH_SIZE = 0,
  DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH = 1,
  DYNAMIC_SHAPE_MODE_N_DIMS = 2,
  DYNAMIC_SHAPE_MODE_SHAPE_RANGE = 3,
} DynamicShapeMode;

typedef struct AscendConfigParams {
  std::string profiling_file_path = "";
  std::string dump_model_path = "";
  std::string precision_mode = "";
  std::string modify_mixlist_path = "";
  std::string op_select_impl_mode = "";
  std::string op_type_list_for_impl_mode = "";
  std::string enable_compress_weight = "";
  std::string auto_tune_mode = "";
  std::string enable_dynamic_shape_range = "";
  int64_t initial_buffer_length_of_dynamic_shape_range = -1;
} AscendConfigParams;

class AclModelClient {
 public:
  explicit AclModelClient(int device_id, AscendConfigParams* config_params);
  ~AclModelClient();

  bool LoadModel(const void* data,
                 size_t size,
                 AscendConfigParams* config_params);
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
  bool CreateModelInputDataset(bool is_dynamic_shape_range,
                               int64_t buffer_length = -1);
  bool CreateModelOutputDataset(bool is_dynamic_shape_range,
                                int64_t buffer_length = -1);
  void DestroyDataset(aclmdlDataset** dataset);
  void ProfilingStart();
  void ProfilingEnd();
  void SetDynamicShapeRangeInitialBufferLength(int64_t initial_buffer_length) {
    dynamic_shape_range_initial_buffer_length_ = initial_buffer_length;
  }

 private:
  int device_id_{0};
  aclrtContext context_{nullptr};
  uint32_t model_id_{0};
  aclmdlDesc* model_desc_{nullptr};
  aclmdlDataset* input_dataset_{nullptr};
  aclmdlDataset* output_dataset_{nullptr};
  aclprofConfig* config_{nullptr};
  int64_t dynamic_shape_range_initial_buffer_length_{4 * 3 * 1024 * 1024};
};

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
