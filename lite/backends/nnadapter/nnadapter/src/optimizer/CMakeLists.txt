# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(nnadapter_optimizer_pattern_matcher STATIC pattern_matcher.cc)
add_library(nnadapter_optimizer_convert_quantization_symm_to_asymm STATIC convert_quantization_symm_to_asymm.cc)
add_library(nnadapter_optimizer_convert_datalayout_nchw_to_nhwc STATIC convert_datalayout_nchw_to_nhwc.cc)
add_library(nnadapter_optimizer_fuse_matmul_add_into_fully_connected STATIC fuse_matmul_add_into_fully_connected.cc)
add_library(nnadapter_optimizer_partition_model_into_submodels STATIC partition_model_into_submodels.cc)
add_library(nnadapter_optimizer_constant_fold_shape_and_associated_operations STATIC constant_fold_shape_and_associated_operations.cc)

set(NNADAPTER_OPTIMIZERS nnadapter_optimizer_pattern_matcher nnadapter_optimizer_convert_quantization_symm_to_asymm nnadapter_optimizer_convert_datalayout_nchw_to_nhwc nnadapter_optimizer_fuse_matmul_add_into_fully_connected nnadapter_optimizer_partition_model_into_submodels nnadapter_optimizer_constant_fold_shape_and_associated_operations CACHE INTERNAL "")
