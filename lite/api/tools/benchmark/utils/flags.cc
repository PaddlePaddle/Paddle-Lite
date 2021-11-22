// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/api/tools/benchmark/utils/flags.h"

namespace paddle {
namespace lite_api {

// Model options
DEFINE_string(optimized_model_file, "", optimized_model_file_msg);
DEFINE_string(uncombined_model_dir, "", uncombined_model_dir_msg);
DEFINE_string(model_file, "", model_file_msg);
DEFINE_string(param_file, "", param_file_msg);
DEFINE_string(input_shape, "", input_shape_msg);
DEFINE_string(input_data_path, "", input_data_path_msg);
DEFINE_string(validation_set, "", validation_set_msg);
DEFINE_bool(show_output_elem, false, show_output_elem_msg);

// Common runtime options
DEFINE_int32(warmup, 0, warmup_msg);
DEFINE_int32(repeats, 1, repeats_msg);
DEFINE_double(run_delay, -1.0, run_delay_msg);
DEFINE_int32(power_mode, 0, power_mode_msg);
DEFINE_int32(threads, 1, threads_msg);
DEFINE_string(result_path, "", result_path_msg);

// Backend options
DEFINE_string(backend, "", backend_msg);
DEFINE_string(cpu_precision, "fp32", cpu_precision_msg);
DEFINE_string(gpu_precision, "fp16", gpu_precision_msg);
DEFINE_string(opencl_cache_dir, "", opencl_cache_dir_msg);
DEFINE_string(opencl_kernel_cache_file,
              "paddle_lite_opencl_kernel.bin",
              opencl_kernel_cache_file_msg);
DEFINE_string(opencl_tuned_file,
              "paddle_lite_opencl_tuned.params",
              opencl_tuned_file_msg);
DEFINE_string(opencl_tune_mode, "normal", opencl_tune_mode_msg);
DEFINE_string(nnadapter_device_names, "", nnadapter_device_names_msg);
DEFINE_string(nnadapter_context_properties,
              "",
              nnadapter_context_properties_msg);

// Profiling options
DEFINE_bool(enable_op_time_profile, false, enable_op_time_profile_msg);
DEFINE_bool(enable_memory_profile, false, enable_memory_profile_msg);
DEFINE_int32(memory_check_interval_ms, 5, memory_check_interval_ms_msg);

// Configuration options
DEFINE_string(config_path, "", config_path_msg);

// Others

}  // namespace lite_api
}  // namespace paddle
