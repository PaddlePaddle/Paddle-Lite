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

#include <gflags/gflags.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/tools/opt_base.h"

DEFINE_string(model_dir,
              "",
              "path of the model. This option will be ignored if model_file "
              "and param_file are exist");
DEFINE_string(model_filename,
              "",
              "model topo filename of the model in models set. This option"
              " will be used to specific tailoring");
DEFINE_string(param_filename,
              "",
              "model param filename of the model in models set. This option"
              " will be used to specific tailoring");
DEFINE_string(model_set_dir,
              "",
              "path of the models set. This option will be used to specific"
              " tailoring");
DEFINE_string(model_file, "", "model file path of the combined-param model");
DEFINE_string(param_file, "", "param file path of the combined-param model");
DEFINE_string(
    optimize_out_type,
    "naive_buffer",
    "store type of the output optimized model. protobuf/naive_buffer");
DEFINE_bool(display_kernels, false, "Display kernel information");
DEFINE_bool(quant_model,
            false,
            "Use post_quant_dynamic method to quantize the model weights.");
DEFINE_string(quant_type,
              "QUANT_INT16",
              "Set the quant_type for post_quant_dynamic, "
              "and it should be QUANT_INT8 or QUANT_INT16 for now.");
DEFINE_bool(enable_fp16, false, "Set kernel_type run in FP16.");
DEFINE_bool(record_tailoring_info,
            false,
            "Record kernels and operators information of the optimized model "
            "for tailoring compiling, information are stored into optimized "
            "model path as hidden files");
DEFINE_string(optimize_out, "", "path of the output optimized model");
DEFINE_string(valid_targets,
              "arm",
              "The targets this model optimized for, should be one of (arm, "
              "opencl, x86, x86_opencl), splitted by space");
DEFINE_bool(print_supported_ops,
            false,
            "Print supported operators on the inputed target");
DEFINE_bool(print_all_ops,
            false,
            "Print all the valid operators of Paddle-Lite");
DEFINE_bool(print_all_ops_in_md_format,
            false,
            "Print all the valid operators of Paddle-Lite to modify docs");
DEFINE_bool(print_model_ops, false, "Print operators in the input model");
DEFINE_bool(sparse_model,
            false,
            "Use sparse_conv_detect_pass to sparsify the 1x1conv weights.");
DEFINE_double(sparse_threshold,
              0.6,
              "Set 0.6 as the lower bound for the sparse conv pass.");
DEFINE_string(optimized_nb_model_path,
              "",
              "path of the optimized nb model, this argument is use for the "
              "VisualizeOptimizedModel API");
DEFINE_string(visualization_file_output_path,
              "",
              "output path of the visualization file, this argument is use for "
              "the VisualizeOptimizedModel API");
DEFINE_string(nnadapter_mixed_precision_quantization_config_path,
              "",
              "path of nnadapter mixed precision quantization config, this "
              "argument is used to remove quant info of ops");

int main(int argc, char** argv) {
  auto opt = paddle::lite_api::OptBase();
  // If there is none input argument, print help info.
  if (argc < 2) {
    opt.PrintExecutableBinHelpInfo();
    return 0;
  }
  google::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_model_dir != "") {
    opt.SetModelDir(FLAGS_model_dir);
  }
  if (FLAGS_model_set_dir != "") {
    opt.SetModelSetDir(FLAGS_model_set_dir);
  }
  if (FLAGS_model_file != "") {
    opt.SetModelFile(FLAGS_model_file);
  }
  if (FLAGS_param_file != "") {
    opt.SetParamFile(FLAGS_param_file);
  }
  if (FLAGS_optimize_out_type != "") {
    opt.SetModelType(FLAGS_optimize_out_type);
  }
  if (FLAGS_optimize_out != "") {
    opt.SetOptimizeOut(FLAGS_optimize_out);
  }
  if (FLAGS_valid_targets != "") {
    if (FLAGS_enable_fp16) opt.EnableFloat16();
    opt.SetValidPlaces(FLAGS_valid_targets);
  }
  if (FLAGS_nnadapter_mixed_precision_quantization_config_path != "") {
    opt.SetNNAdapterMixedPrecisionQuantizationConfigPath(
        FLAGS_nnadapter_mixed_precision_quantization_config_path);
  }

  if (FLAGS_record_tailoring_info) {
    opt.RecordModelInfo(true);
  }
  if (FLAGS_quant_model) {
    opt.SetQuantModel(true);
    opt.SetQuantType(FLAGS_quant_type);
  }
  if (FLAGS_sparse_model) {
    opt.SetSparseModel(true);
    opt.SetSparseThreshold(FLAGS_sparse_threshold);
  }
  if (FLAGS_print_all_ops) {
    opt.PrintAllOps();
    return 0;
  }
  if (FLAGS_print_supported_ops) {
    opt.PrintSupportedOps();
    return 0;
  }
  if (FLAGS_display_kernels) {
    opt.DisplayKernelsInfo();
    return 0;
  }
  if (FLAGS_print_model_ops) {
    opt.CheckIfModelSupported(true);
    return 0;
  }
  if (FLAGS_print_all_ops_in_md_format) {
    opt.PrintAllSupportedOpsInMdformat();
    return 0;
  }
  if (FLAGS_optimized_nb_model_path != "" &&
      FLAGS_visualization_file_output_path != "") {
    opt.VisualizeOptimizedNBModel(FLAGS_optimized_nb_model_path,
                                  FLAGS_visualization_file_output_path);
    return 0;
  }
  if ((FLAGS_model_dir == "" &&
       (FLAGS_model_file == "" || FLAGS_param_file == "") &&
       FLAGS_model_set_dir == "") ||
      FLAGS_optimize_out == "") {
    opt.PrintExecutableBinHelpInfo();
    return 1;
  }

  opt.Run();
  return 0;
}
