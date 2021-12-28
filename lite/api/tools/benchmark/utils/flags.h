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

#ifndef LITE_API_TOOLS_BENCHMARK_UTILS_FLAGS_H_
#define LITE_API_TOOLS_BENCHMARK_UTILS_FLAGS_H_
#include <gflags/gflags.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h

namespace paddle {
namespace lite_api {

// Model options
static const char optimized_model_file_msg[] =
    "The filename of the model that is optimized by opt.";
static const char uncombined_model_dir_msg[] =
    "The dir of the uncombined model, the model and param files "
    "are under model_dir.";
static const char model_file_msg[] =
    "The filename of model file. Set model_file when the model is "
    "combined format.";
static const char param_file_msg[] =
    "The filename of param file. Set param_file when the model is "
    "combined format.";
static const char input_shape_msg[] =
    "Set input shapes according to the model, "
    "separated by comma and colon, "
    "such as 1,3,224,224 for only one input, "
    "1,3,224,224:1,5 for two inputs.";
static const char input_data_path_msg[] =
    "Set the path(s) of model input, "
    "separated by comma and colon, "
    "such as /path/to/in.txt for only one input, "
    "/path/to/in0.txt:/path/to/in1.txt for two inputs."
    "The input of model will be 1.0 if this option in not set.";
static const char validation_set_msg[] =
    "Use validation images and lables as inputs. Only supports a minival "
    "dataset of ILSVRC_2012 as inputs."
    "Supported set: ILSVRC_2012";
static const char show_output_elem_msg[] =
    "Show each output tensor's all elements.";

// Common runtime options
static const char warmup_msg[] = "warmup times";
static const char repeats_msg[] = "repeats times";
static const char run_delay_msg[] =
    "The delay in seconds between subsequent benchmark runs. "
    "Non-positive values mean use no delay.";
static const char power_mode_msg[] =
    "arm power mode: "
    "0 for big cluster, "
    "1 for little cluster, "
    "2 for all cores, "
    "3 for no bind";
static const char threads_msg[] = "threads num";
static const char result_path_msg[] = "Save benchmark info to the file.";

// Backend options
static const char backend_msg[] =
    "To use a particular backend for execution. "
    "Should be one of: arm|opencl|x86|x86_opencl|"
    "npu|xpu|nnadapter|";
static const char cpu_precision_msg[] =
    "Register fp32 or fp16 arm-cpu kernel when optimized model. "
    "Should be one of: fp32, fp16.";
static const char gpu_precision_msg[] =
    "Set precision of computation in GPU. "
    "Should be one of: fp32, fp16.";
static const char opencl_cache_dir_msg[] =
    "A directory in which kernel binary and tuned file will be stored."
    "Kernel cache & tune both are OFF if --opencl_cache_dir is not set.";
static const char opencl_kernel_cache_file_msg[] =
    "Set opencl kernel binary filename. "
    "We strongly recommend each model has a unique binary name.";
static const char opencl_tuned_file_msg[] =
    "Set opencl tuned filename."
    "We strongly recommend each model has a unique param name.";
static const char opencl_tune_mode_msg[] =
    "Set opencl tune option: none, rapid, normal, exhaustive.";
static const char nnadapter_device_names_msg[] =
    "Set nnadapter device names. "
    "Should be one of: huawei_kirin_npu|huawei_ascend_npu|rockchip_npu|"
    "imagination_nna|mediatek_apu|amlogic_npu|verisilicon_timvx|";
static const char nnadapter_context_properties_msg[] =
    "Set nnadapter device hardware resources, default to null";

// Profiling options
static const char enable_op_time_profile_msg[] =
    "Whether to run with op time profiling. Not supported yet";
static const char enable_memory_profile_msg[] =
    "Whether to report the memory usage by periodically "
    "checking the memory footprint. Internally, a separate thread "
    " will be spawned for this periodic check. Therefore, "
    "the performance benchmark result could be affected. Not supported yet.";
static const char memory_check_interval_ms_msg[] =
    "The interval in millisecond between two consecutive memory "
    "footprint checks. This is only used when "
    "--enable_memory_profile is set to true. Not supported yet.";

// Configuration options
static const char config_path_msg[] = "Configuration options.";

// Others

// Model options
DECLARE_string(optimized_model_file);
DECLARE_string(uncombined_model_dir);
DECLARE_string(model_file);
DECLARE_string(param_file);
DECLARE_string(input_shape);
DECLARE_string(input_data_path);
DECLARE_string(validation_set);
DECLARE_bool(show_output_elem);

// Common runtime options
DECLARE_int32(warmup);
DECLARE_int32(repeats);
DECLARE_double(run_delay);
DECLARE_int32(power_mode);
DECLARE_int32(threads);
DECLARE_string(result_path);

// Backend options
DECLARE_string(backend);
DECLARE_string(cpu_precision);
DECLARE_string(gpu_precision);
DECLARE_string(opencl_cache_dir);
DECLARE_string(opencl_kernel_cache_file);
DECLARE_string(opencl_tuned_file);
DECLARE_string(opencl_tune_mode);
DECLARE_string(nnadapter_device_names);
DECLARE_string(nnadapter_context_properties);

// Profiling options
DECLARE_bool(enable_op_time_profile);
DECLARE_bool(enable_memory_profile);
DECLARE_int32(memory_check_interval_ms);

// Configuration options
DECLARE_string(config_path);

// Others

}  // namespace lite_api
}  // namespace paddle

#endif  // LITE_API_TOOLS_BENCHMARK_UTILS_FLAGS_H_
