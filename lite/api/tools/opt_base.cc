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

#include "lite/api/tools/opt_base.h"
#include <fstream>
#include <memory>
#include <utility>
#include "lite/core/optimizer/mir/dot.h"
#include "lite/core/scope.h"
#include "lite/utils/string.h"
namespace paddle {
namespace lite_api {

void OptBase::SetModelDir(const std::string& model_path) {
  opt_config_.set_model_dir(model_path);
}

void OptBase::SetModelFile(const std::string& model_path) {
  opt_config_.set_model_file(model_path);
}

void OptBase::SetParamFile(const std::string& param_path) {
  opt_config_.set_param_file(param_path);
}

void OptBase::SetModelType(std::string optimize_out_type) {
  if (optimize_out_type == "protobuf") {
    model_type_ = LiteModelType::kProtobuf;
  } else if (optimize_out_type == "naive_buffer") {
    model_type_ = LiteModelType::kNaiveBuffer;
  } else {
    OPT_LOG_FATAL << "Unsupported Model type :" << optimize_out_type;
  }
}

void OptBase::SetQuantModel(bool quant_model) {
  opt_config_.set_quant_model(quant_model);
}

void OptBase::SetQuantType(const std::string& quant_type) {
  if (quant_type == "QUANT_INT8") {
    opt_config_.set_quant_type(lite_api::QuantType::QUANT_INT8);
  } else if (quant_type == "QUANT_INT16") {
    opt_config_.set_quant_type(lite_api::QuantType::QUANT_INT16);
  } else {
    OPT_LOG_FATAL << "Unsupported quant type: " << quant_type;
  }
}

void OptBase::SetSparseModel(bool sparse_model) {
  opt_config_.set_sparse_model(sparse_model);
}

void OptBase::SetSparseThreshold(float sparse_threshold) {
  // sparse_model mode only supported on Arm.
  TargetType target;
  for (size_t i = 0; i < valid_places_.size(); i++) {
    target = valid_places_[i].target;
    if (target != TargetType::kARM) {
      OPT_LOG << "sparse_model mode only supported on Arm. The model will "
                 "be optimized to dense format.";
      opt_config_.set_sparse_model(false);
      break;
    }
  }
  // threshold must be between 0 and 1.
  if (sparse_threshold < 0.0 || sparse_threshold > 1.0) {
    OPT_LOG_FATAL << "Please set sparse_threshold between 0.0 and 1.0.";
  } else {
    opt_config_.set_sparse_threshold(sparse_threshold);
  }
}

void OptBase::SetNNAdapterMixedPrecisionQuantizationConfigPath(
    const std::string& nnadapter_mixed_precision_quantization_config_path) {
  opt_config_.set_nnadapter_mixed_precision_quantization_config_path(
      nnadapter_mixed_precision_quantization_config_path);
}

void OptBase::SetPassesInternal(
    const std::vector<std::string>& passes_internal) {
  opt_config_.set_passes_internal(passes_internal);
}

void OptBase::SetValidPlaces(const std::string& valid_places) {
  valid_places_.clear();
  auto target_reprs = lite::Split(valid_places, ",");
  std::vector<std::string> nnadapter_device_names;
  for (auto& target_repr : target_reprs) {
    if (target_repr == "arm") {
      if (enable_fp16_) {
        valid_places_.emplace_back(
            Place{TARGET(kARM), PRECISION(kFP16), DATALAYOUT(kNCHW)});
      }
      valid_places_.emplace_back(
          Place{TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kARM), PRECISION(kInt32), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kARM), PRECISION(kInt64), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kARM), PRECISION(kAny), DATALAYOUT(kNCHW)});
    } else if (target_repr == "opencl") {
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageFolder)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          TARGET(kARM));  // enable kARM CPU kernel when no opencl kernel
    } else if (target_repr == "metal") {
      valid_places_.emplace_back(Place{
          TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)});
      valid_places_.emplace_back(Place{
          TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)});
    } else if (target_repr == "arm_metal") {
      valid_places_.emplace_back(Place{
          TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)});
      valid_places_.emplace_back(Place{
          TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)});
      valid_places_.emplace_back(TARGET(kARM));
      valid_places_.emplace_back(TARGET(kHost));
    } else if (target_repr == "x86_metal") {
      valid_places_.emplace_back(Place{
          TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)});
      valid_places_.emplace_back(Place{
          TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)});
      valid_places_.emplace_back(TARGET(kX86));
      valid_places_.emplace_back(TARGET(kHost));
    } else if (target_repr == "x86") {
      valid_places_.emplace_back(Place{TARGET(kX86), PRECISION(kFloat)});
      valid_places_.emplace_back(Place{TARGET(kX86), PRECISION(kInt64)});
      valid_places_.emplace_back(Place{TARGET(kX86), PRECISION(kAny)});
    } else if (target_repr == "x86_opencl") {
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageFolder)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(Place{TARGET(kX86), PRECISION(kFloat)});
      valid_places_.emplace_back(Place{TARGET(kX86), PRECISION(kInt64)});
    } else if (target_repr == "xpu") {
      valid_places_.emplace_back(TARGET(kXPU));
    } else if (target_repr == "mlu") {
      valid_places_.emplace_back(TARGET(kMLU));
    } else if (target_repr == "bm") {
      valid_places_.emplace_back(TARGET(kBM));
    } else if (target_repr == "imagination_nna") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kInt8), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "intel_fpga") {
      valid_places_.emplace_back(TARGET(kIntelFPGA));
      valid_places_.emplace_back(
          Place{TARGET(kIntelFPGA), PRECISION(kFloat), DATALAYOUT(kNCHW)});
    } else if (target_repr == "rockchip_npu") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kInt8), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "mediatek_apu") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kInt8), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "huawei_kirin_npu") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "huawei_ascend_npu") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "amlogic_npu") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "verisilicon_timvx") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "eeasytech_npu") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "android_nnapi") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "qualcomm_qnn") {
      valid_places_.emplace_back(TARGET(kNNAdapter));
      valid_places_.emplace_back(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
      nnadapter_device_names.push_back(target_repr);
    } else if (target_repr == "host") {
      valid_places_.emplace_back(TARGET(kHost));
    } else {
      OPT_LOG_FATAL << lite::string_format(
          "Wrong target '%s' found, please check the command flag "
          "'valid_targets'",
          target_repr.c_str());
    }
  }
  CHECK(!valid_places_.empty())
      << "At least one target should be set, should set the "
         "command argument 'valid_targets'";
  if (!nnadapter_device_names.empty()) {
    opt_config_.set_nnadapter_device_names(nnadapter_device_names);
  }
}

void OptBase::SetOptimizeOut(const std::string& lite_out_name) {
  lite_out_name_ = lite_out_name;
}

void OptBase::RecordModelInfo(bool record_strip_info) {
  record_strip_info_ = record_strip_info;
}

void OptBase::Run() {
  CheckIfModelSupported(false);
  OpKernelInfoCollector::Global().SetKernel2path(kernel2path_map);
  opt_config_.set_valid_places(valid_places_);
  if (model_set_dir_ != "") {
    RunOptimizeFromModelSet(record_strip_info_);
  } else {
    auto opt_predictor = lite_api::CreatePaddlePredictor(opt_config_);
    opt_predictor->SaveOptimizedModel(
        lite_out_name_, model_type_, record_strip_info_);
  }
}

void OptBase::RunOptimize(const std::string& model_dir_path,
                          const std::string& model_path,
                          const std::string& param_path,
                          const std::string& model_type,
                          const std::string& valid_places,
                          const std::string& optimized_out_path) {
  SetModelDir(model_dir_path);
  SetModelFile(model_path);
  SetParamFile(param_path);
  SetModelType(model_type);
  SetValidPlaces(valid_places);
  SetOptimizeOut(optimized_out_path);
  CheckIfModelSupported(false);
  OpKernelInfoCollector::Global().SetKernel2path(kernel2path_map);
  opt_config_.set_valid_places(valid_places_);
  if (model_set_dir_ != "") {
    RunOptimizeFromModelSet(record_strip_info_);
  } else {
    auto opt_predictor = lite_api::CreatePaddlePredictor(opt_config_);
    opt_predictor->SaveOptimizedModel(
        lite_out_name_, model_type_, record_strip_info_);
  }
}
// collect ops info of modelset
void CollectModelMetaInfo(const std::string& output_dir,
                          const std::vector<std::string>& models,
                          const std::string& filename) {
  std::set<std::string> total;
  for (const auto& name : models) {
    std::string model_path =
        lite::Join<std::string>({output_dir, name, filename}, "/");
    auto lines = lite::ReadLines(model_path);
    total.insert(lines.begin(), lines.end());
  }
  std::string output_path =
      lite::Join<std::string>({output_dir, filename}, "/");
  lite::WriteLines(std::vector<std::string>(total.begin(), total.end()),
                   output_path);
}

void OptBase::SetModelSetDir(const std::string& model_set_path) {
  model_set_dir_ = model_set_path;
}
void OptBase::RunOptimizeFromModelSet(bool record_strip_info) {
  // 1. mkdir of outputed optimized model set.
  lite::MkDirRecur(lite_out_name_);
  auto model_dirs = lite::ListDir(model_set_dir_, true);
  if (model_dirs.size() == 0) {
    OPT_LOG_FATAL << "[" << model_set_dir_ << "] does not contain any model";
  }

  // 2. optimize each model in inputed model set dir.
  std::string model_file = opt_config_.model_file();
  std::string param_file = opt_config_.param_file();
  for (const auto& name : model_dirs) {
    std::string input_model_dir =
        lite::Join<std::string>({model_set_dir_, name}, "/");
    std::string output_model_dir =
        lite::Join<std::string>({lite_out_name_, name}, "/");

    if (opt_config_.model_file() != "" && opt_config_.param_file() != "") {
      auto model_file_path =
          lite::Join<std::string>({input_model_dir, model_file}, "/");
      auto param_file_path =
          lite::Join<std::string>({input_model_dir, param_file}, "/");
    }

    opt_config_.set_model_dir(input_model_dir);
    opt_config_.set_model_file(model_file);
    opt_config_.set_param_file(param_file);

    auto opt_predictor = lite_api::CreatePaddlePredictor(opt_config_);
    opt_predictor->SaveOptimizedModel(
        lite_out_name_, model_type_, record_strip_info);
  }

  // 3. if record_strip_info = true, we will record striping info
  if (record_strip_info) {
    // Collect all models information
    CollectModelMetaInfo(
        lite_out_name_, model_dirs, lite::TAILORD_OPS_SOURCE_LIST_FILENAME);
    CollectModelMetaInfo(
        lite_out_name_, model_dirs, lite::TAILORD_OPS_LIST_NAME);
    CollectModelMetaInfo(
        lite_out_name_, model_dirs, lite::TAILORD_KERNELS_SOURCE_LIST_FILENAME);
    CollectModelMetaInfo(
        lite_out_name_, model_dirs, lite::TAILORD_KERNELS_LIST_NAME);
    OPT_LOG << "Record the information of stripped models into :"
            << lite_out_name_ << "successfully";
  }
}

void OptBase::PrintHelpInfo() {
  const std::string opt_version = lite::version();
  const char help_info[] =
      "------------------------------------------------------------------------"
      "-----------------------------------------------------------\n"
      "  Valid arguments of Paddle-Lite opt are listed below:\n"
      "------------------------------------------------------------------------"
      "-----------------------------------------------------------\n"
      "  Arguments of help information:\n"
      "        `help()`   Print help infomation\n"
      "\n"
      "  Arguments of model transformation:\n"
      "        `set_model_dir(model_dir)`\n"
      "        `set_model_file(model_file_path)`\n"
      "        `set_param_file(param_file_path)`\n"
      "        `set_model_type(protobuf|naive_buffer)`: naive_buffer by "
      "default\n"
      "        `set_lite_out(output_optimize_model_dir)`\n"
      "        "
      "`set_valid_places(arm|opencl|x86|metal|xpu|bm|mlu|intel_fpga|"
      "huawei_ascend_npu|imagination_nna|rockchip_npu|"
      "mediatek_apu|huawei_kirin_npu|amlogic_npu|verisilicon_timvx|"
      "eeasytech_npu|android_nnapi|qualcomm_qnn)`"
      "\n"
      "        `record_model_info(false|true)`: refer to whether to record ops "
      "info for striping lib, false by default`\n"
      "        `run() : start model transformation`\n"
      "    eg. `opt.set_model_dir(\"./mobilenetv1\"); "
      "opt.set_lite_out(\"mobilenetv1_opt\"); opt.set_valid_places(\"arm\"); "
      "opt.run();`\n"
      "\n"
      "  You can also transform model through a single input argument:\n"
      "        `run_optimize(model_dir, model_file_path, param_file_path, "
      "model_type, valid_places, lite_out_name) `\n"
      "    eg. `opt.run_optimize(\"./mobilenetv1\", \"\", \"\", "
      "\"naive_buffer\", \"arm\", \"mobilenetv1_opt\");`"
      "\n"
      "  Arguments of checking model and printing ops information:\n"
      "        `print_all_ops()`   Display all the valid operators of "
      "Paddle-Lite\n"
      "        `print_supported_ops`   Display supported operators of valid "
      "places\n"
      "        `check_if_model_supported()`   Check if the input model is "
      "supported\n"
      "  How to print detailed information: export GLOG_v=1 \n"
      "------------------------------------------------------------------------"
      "-----------------------------------------------------------\n";
  OPT_LOG << "opt version:" << opt_version;
  OPT_LOG << help_info;
  exit(1);
}

void OptBase::PrintExecutableBinHelpInfo() {
  const std::string opt_version = lite::version();
  const char help_info[] =
      "At least one argument should be inputed. Valid arguments are listed "
      "below:\n"
      "  Arguments of model optimization:\n"
      "        `--model_dir=<model_param_dir>`\n"
      "        `--model_file=<model_path>`\n"
      "        `--param_file=<param_path>`\n"
      "        `--optimize_out_type=(protobuf|naive_buffer)`\n"
      "        `--optimize_out=<output_optimize_model_dir>`\n"
      "        "
      "`--valid_targets=(arm|opencl|x86|metal|xpu|bm|mlu|intel_fpga|"
      "huawei_ascend_npu|imagination_nna|rockchip_npu|mediatek_apu|"
      "huawei_kirin_npu|amlogic_npu|verisilicon_timvx|android_nnapi|"
      "qualcomm_qnn)`\n"
      "        `--record_tailoring_info=(true|false)`\n"
      "  Arguments of mode quantization in opt:\n"
      "        `--quant_model=(true|false)`\n"
      "        `--quant_type=(QUANT_INT8|QUANT_INT16)`\n"
      "  Arguements of sparse convolution in opt: \n"
      "        `--sparse_model=(true|false)`\n"
      "        `--sparse_threshold=(float)`\n"
      "  Arguments of enable_fp16 in opt: \n"
      "        `--enable_fp16=(true|false)`\n"
      "  Arguments of model checking and ops information:\n"
      "        `--print_all_ops=true`   Display all the valid operators of "
      "Paddle-Lite\n"
      "        `--print_all_ops_in_md_format=true`   Display all the valid "
      "operators of "
      "Paddle-Lite in markdown format\n"
      "        `--print_supported_ops=true  "
      "--valid_targets=(arm|opencl|x86|metal|xpu|bm|mlu|intel_fpga|"
      "huawei_ascend_npu|imagination_nna|rockchip_npu|mediatek_apu|"
      "huawei_kirin_npu|amlogic_npu|verisilicon_timvx|android_nnapi|"
      "qualcomm_qnn)`"
      "  Display valid operators of input targets\n"
      "        `--print_model_ops=true  --model_dir=<model_param_dir> "
      "--valid_targets=(arm|opencl|x86|metal|xpu|bm|mlu|intel_fpga|"
      "huawei_ascend_npu|imagination_nna|rockchip_npu|mediatek_apu|"
      "huawei_kirin_npu|amlogic_npu|verisilicon_timvx|android_nnapi|"
      "qualcomm_qnn)`"
      "  Display operators in the input model\n"
      "  Arguments of optimized nb model visualization: \n"
      "        `--optimized_nb_model_path=<optimized_nb_model_dir>`\n"
      "        "
      "`--visualization_file_output_path=<visualization_file_output_path>`\n";
  OPT_LOG << "paddlelite opt version:" << opt_version;
  OPT_LOG << help_info;
}

// 2. Print supported info of inputed ops
void OptBase::PrintOpsInfo(const std::set<std::string>& valid_ops,
                           const std::vector<std::string> valid_targets) {
  // Get the lengh of the first column: maximum length of the op_type
  size_t maximum_optype_length = 0;
  for (auto it = all_supported_ops_.begin(); it != all_supported_ops_.end();
       it++) {
    maximum_optype_length = it->first.size() > maximum_optype_length
                                ? it->first.size()
                                : maximum_optype_length;
  }
  // Print the first row: OP_nam taget1 target2 ...
  std::cout << std::setw(maximum_optype_length) << "OP_name";
  for (size_t i = 0; i < valid_targets.size(); i++) {
    std::cout << std::setw(valid_targets[i].size() + 2) << valid_targets[i];
  }
  std::cout << std::endl;
  // Print the name of supported ops and mark if it's supported by each target
  // print the support info of inputed ops: valid_ops
  for (auto op = valid_ops.begin(); op != valid_ops.end(); op++) {
    // Check: If this kernel doesn't match any operator, we will skip it.
    if (all_supported_ops_.find(*op) == all_supported_ops_.end()) {
      continue;
    }
    std::cout << std::setw(maximum_optype_length) << *op;
    // Print OP info.
    auto ops_valid_places = all_supported_ops_.at(*op);
    for (size_t i = 0; i < valid_targets.size(); i++) {
      if (std::find(ops_valid_places.begin(),
                    ops_valid_places.end(),
                    valid_targets[i]) != ops_valid_places.end()) {
        std::cout << std::setw(valid_targets[i].size() + 2) << "Y";
      } else {
        std::cout << std::setw(valid_targets[i].size() + 2) << " ";
      }
    }
    std::cout << std::endl;
  }
}

void OptBase::DisplayKernelsInfo() {  // Display kernel information
  OPT_LOG << ::paddle::lite::KernelRegistry::Global().DebugString();
}
void OptBase::PrintAllOps() {
  // 1. Get all supported ops
  std::set<std::string> valid_ops;
  for (auto& elem : target_supported_ops_) {
    auto ops = elem.second;
    valid_ops.insert(ops.begin(), ops.end());
  }
  // 2. Print support info of these ops
  PrintOpsInfo(valid_ops);
}

void OptBase::PrintAllSupportedOpsInMdformat() {
  // 1. Get all supported ops
  std::set<std::string> valid_ops;
  for (auto& elem : target_supported_ops_) {
    valid_ops.insert(elem.second.begin(), elem.second.end());
  }
  std::cout << "The number of supported operators is : " << supported_ops.size()
            << "\n";
  const std::vector<std::string> valid_targets = {"kARM",
                                                  "kOpenCL",
                                                  "kMetal",
                                                  "kXPU",
                                                  "kHost",
                                                  "kX86",
                                                  "kBM",
                                                  "kMLU",
                                                  "kIntelFPGA",
                                                  "huawei_ascend_npu",
                                                  "mediatek_apu",
                                                  "rockchip_npu",
                                                  "huawei_kirin_npu",
                                                  "imagination_nna",
                                                  "amlogic_npu",
                                                  "verisilicon_timvx",
                                                  "eeasytech_npu",
                                                  "android_nnapi",
                                                  "qualcomm_qnn"};
  const std::vector<std::string> readable_valid_targets = {"ARM",
                                                           "OpenCL",
                                                           "Metal",
                                                           "百度XPU",
                                                           "Host",
                                                           "X86",
                                                           "比特大陆NPU",
                                                           "寒武纪MLU",
                                                           "英特尔FPGA",
                                                           "华为昇腾NPU",
                                                           "联发科APU",
                                                           "瑞芯微NPU",
                                                           "华为麒麟NPU",
                                                           "颖脉NNA",
                                                           "晶晨NPU",
                                                           "TIM-VX",
                                                           "亿智NPU",
                                                           "Android NNAPI",
                                                           "高通QNN"};
  // Print the first row: OP_nam taget1 target2 ...
  std::cout << "| "
            << "OP_name ";
  for (size_t i = 0; i < readable_valid_targets.size(); i++) {
    std::cout << "| " << readable_valid_targets[i] << " ";
  }
  std::cout << "\n";

  // Print the second row
  std::cout << "|-:|";
  for (size_t i = 0; i < readable_valid_targets.size(); i++) {
    std::cout << "-|"
              << " ";
  }
  std::cout << "\n";

  // Print the name of supported ops and mark if it's supported by each target
  // print the support info of inputed ops: valid_ops
  for (auto op = valid_ops.begin(); op != valid_ops.end(); op++) {
    // Check: If this kernel doesn't match any operator, we will skip it.
    if (all_supported_ops_.find(*op) == all_supported_ops_.end()) {
      continue;
    }
    std::cout << "| " << *op << " ";
    // Print OP info.
    auto ops_valid_places = all_supported_ops_.at(*op);
    for (size_t i = 0; i < valid_targets.size(); i++) {
      if (std::find(ops_valid_places.begin(),
                    ops_valid_places.end(),
                    valid_targets[i]) != ops_valid_places.end()) {
        std::cout << "| "
                  << "Y ";
      } else {
        std::cout << "|   ";
      }
    }
    std::cout << "|\n";
  }
}

void OptBase::PrintSupportedOps() {
  // 1. Get the valid hardware targets
  std::set<std::string> valid_targets = {};
  for (size_t i = 0; i < valid_places_.size(); i++) {
    std::string target = TargetRepr(valid_places_[i].target);
    if (target == "kNNAdapter") {
      CHECK(opt_config_.nnadapter_device_names().size());
      for (auto& device : opt_config_.nnadapter_device_names())
        valid_targets.insert(device);
    } else {
      valid_targets.insert(target);
    }
  }
  std::string targets_str{};
  for (auto& target : valid_targets) {
    targets_str = targets_str + " " + target;
  }
  OPT_LOG << "Supported OPs on '" << targets_str << "': ";
  valid_targets.insert(TargetRepr(TARGET(kHost)));
  valid_targets.insert(TargetRepr(TARGET(kUnk)));

  // 2. Get supported ops on these targets
  std::set<std::string> valid_ops;
  for (auto& target : valid_targets) {
    auto ops = target_supported_ops_.at(target);
    valid_ops.insert(ops.begin(), ops.end());
  }

  PrintOpsInfo(
      valid_ops,
      std::vector<std::string>(valid_targets.begin(), valid_targets.end()));
}

// test whether this model is supported
void OptBase::CheckIfModelSupported(bool print_ops_info) {
  // 1. parse valid places and valid targets
  auto valid_ops = target_supported_ops_.at("kHost");
  auto valid_unktype_ops = target_supported_ops_.at("kUnk");
  valid_ops.insert(valid_unktype_ops.begin(), valid_unktype_ops.end());
  for (size_t i = 0; i < valid_places_.size(); i++) {
    std::string target = TargetRepr(valid_places_[i].target);
    // get valid ops
    if (target == "kNNAdapter") {
      CHECK(opt_config_.nnadapter_device_names().size());
      for (auto& device : opt_config_.nnadapter_device_names()) {
        auto ops = target_supported_ops_.at(device);
        valid_ops.insert(ops.begin(), ops.end());
      }
    } else {
      auto ops = target_supported_ops_.at(target);
      valid_ops.insert(ops.begin(), ops.end());
    }
  }

  // 2.Load model into program to get ops in model
  bool is_combined_params_form = false;
  if (!(opt_config_.model_file()).empty() &&
      !(opt_config_.param_file()).empty()) {
    is_combined_params_form = true;
  }
  std::string prog_path = lite::FindModelFileName(opt_config_.model_dir(),
                                                  (opt_config_.model_file()),
                                                  is_combined_params_form);

  lite::cpp::ProgramDesc cpp_prog;
  framework::proto::ProgramDesc pb_proto_prog = *lite::LoadProgram(prog_path);
  lite::pb::ProgramDesc pb_prog(&pb_proto_prog);
  // Transform to cpp::ProgramDesc
  lite::TransformProgramDescAnyToCpp(pb_prog, &cpp_prog);

  std::set<std::string> unsupported_ops;
  std::set<std::string> input_model_ops;
  for (size_t index = 0; index < cpp_prog.BlocksSize(); index++) {
    auto current_block = cpp_prog.GetBlock<lite::cpp::BlockDesc>(index);
    for (size_t i = 0; i < current_block->OpsSize(); ++i) {
      auto& op_desc = *current_block->GetOp<lite::cpp::OpDesc>(i);
      auto op_type = op_desc.Type();
      input_model_ops.insert(op_type);
      if (valid_ops.count(op_type) == 0) {
        unsupported_ops.insert(op_type);
      }
    }
  }
  // 3. Print ops_info of input model and check if this model is supported
  if (print_ops_info) {
    OPT_LOG << "OPs in the input model include:";
    std::set<std::string> valid_targets_set;
    for (auto& it : valid_places_) {
      if (it.target == TargetType::kNNAdapter) {
        CHECK(opt_config_.nnadapter_device_names().size());
        for (auto& device : opt_config_.nnadapter_device_names())
          valid_targets_set.insert(device);
      } else {
        valid_targets_set.insert(TargetRepr(it.target));
      }
    }
    valid_targets_set.insert(TargetRepr(TARGET(kHost)));
    std::vector<std::string> valid_targets(valid_targets_set.begin(),
                                           valid_targets_set.end());
    PrintOpsInfo(input_model_ops, valid_targets);
  }
  if (!unsupported_ops.empty()) {
    std::string unsupported_ops_str = *unsupported_ops.begin();
    for (auto op_str = ++unsupported_ops.begin();
         op_str != unsupported_ops.end();
         op_str++) {
      unsupported_ops_str = unsupported_ops_str + ", " + *op_str;
    }
    std::vector<TargetType> targets = {};
    for (size_t i = 0; i < valid_places_.size(); i++) {
      targets.push_back(valid_places_[i].target);
    }
    std::stable_sort(targets.begin(), targets.end());
    targets.erase(unique(targets.begin(), targets.end()), targets.end());
    std::string targets_str = TargetToStr(targets[0]);
    for (size_t i = 1; i < targets.size(); i++) {
      targets_str = targets_str + "," + TargetToStr(targets[i]);
    }

    OPT_LOG_FATAL << "Error: This model is not supported, because "
                  << unsupported_ops.size() << " ops are not supported on '"
                  << targets_str << "'. These unsupported ops are: '"
                  << unsupported_ops_str << "'.";
  }
  if (print_ops_info) {
    OPT_LOG << "Paddle-Lite supports this model!";
    exit(1);
  }
}

std::vector<std::string> OptBase::VisualizeOptimizedNBModel(
    const std::string& model_dir, const std::string& output_path) {
  // Load naive buffer model
  std::shared_ptr<lite::Scope> scope = std::make_shared<lite::Scope>();
  std::shared_ptr<lite::cpp::ProgramDesc> program =
      std::make_shared<lite::cpp::ProgramDesc>();
  LoadModelNaiveFromFile(model_dir, scope.get(), program.get());
  CHECK(program.get());

  paddle::lite::mir::Dot dot;
  using Attr = paddle::lite::mir::Dot::Attr;
  const std::vector<Attr> op_attrs{Attr("style", "filled"),
                                   Attr("fillcolor", "yellow")};
  const std::vector<Attr> var_attrs{Attr("style", "filled"),
                                    Attr("fillcolor", "gray"),
                                    Attr("shape", "record")};
  const std::vector<Attr> edge_attrs{};

  std::fstream fs;
  std::vector<std::string> res{};
  std::set<std::string> vars{};
  for (size_t block_idx = 0; block_idx < program->BlocksSize(); block_idx++) {
    dot.Clear();
    vars.clear();
    const lite::cpp::BlockDesc* block =
        program->GetBlock<lite::cpp::BlockDesc>(block_idx);
    for (size_t op_idx = 0; op_idx < block->OpsSize(); op_idx++) {
      const lite::cpp::OpDesc* op = block->GetOp<lite::cpp::OpDesc>(op_idx);
      for (auto& in : op->input_vars()) {
        vars.insert(in);
      }
      for (auto& out : op->output_vars()) {
        vars.insert(out);
      }
    }
    for (auto& var : vars) {
      dot.AddNode(var, var_attrs);
    }
    for (size_t op_idx = 0; op_idx < block->OpsSize(); op_idx++) {
      const lite::cpp::OpDesc* op = block->GetOp<lite::cpp::OpDesc>(op_idx);
      const std::string op_indx_str = std::to_string(op_idx);
      dot.AddNode(op_indx_str, op_attrs, op->Type());
      for (auto& var_name : op->input_vars()) {
        dot.AddEdge(var_name, op_indx_str, edge_attrs);
      }
      for (auto& var_name : op->output_vars()) {
        dot.AddEdge(op_indx_str, var_name, edge_attrs);
      }
    }
    std::string graph = dot.Build();

    std::string file_name = "Block_" + std::to_string(block_idx);
    if (output_path.empty())
      LOG(FATAL) << "output_path is empty, please set output_path to save "
                    "visualization file";
    else
      fs.open(output_path + "/" + file_name + ".dot", std::ios::out);
    CHECK(fs.is_open()) << "output path error";
    fs.write(graph.c_str(), graph.size());
    res.emplace_back(std::move(file_name));
    fs.close();
  }
  return res;
}

void OptBase::InitSupportedOpInfo() {
  // collected targets in compile time, which are in head file
  // supported_kernel_op_info.h
  std::vector<std::string> collect_targets = {"kUnk",
                                              "kHost",
                                              "kX86",
                                              "kCUDA",
                                              "kARM",
                                              "kOpenCL",
                                              "kAny",
                                              "kFPGA",
                                              "kNPU",
                                              "kXPU",
                                              "kBM",
                                              "kMLU",
                                              "kRKNPU",
                                              "kIntelFPGA",
                                              "kMetal",
                                              "kNNAdapter"};

  // ignore some old targets
  std::set<std::string> valid_target{"kARM",
                                     "kOpenCL",
                                     "kMetal",
                                     "kXPU",
                                     "kHost",
                                     "kIntelFPGA",
                                     "kX86",
                                     "kBM",
                                     "kMLU",
                                     "huawei_ascend_npu",
                                     "mediatek_apu",
                                     "rockchip_npu",
                                     "huawei_kirin_npu",
                                     "imagination_nna",
                                     "amlogic_npu",
                                     "verisilicon_timvx",
                                     "eeasytech_npu",
                                     "android_nnapi",
                                     "qualcomm_qnn",
                                     "kUnk"};
  for (size_t idx = 0; idx < supported_ops_target.size(); idx++) {
    if (valid_target.find(collect_targets[idx]) != valid_target.end()) {
      auto& support_ops = target_supported_ops_[collect_targets[idx]];
      support_ops.insert(supported_ops_target[idx].begin(),
                         supported_ops_target[idx].end());
    }
  }

  for (auto& elem : supported_ops) {
    for (auto target : elem.second) {
      all_supported_ops_[elem.first].insert(target);
    }
  }

  // collect operators supported by nnadapter
  // operators in head file converter/all.h
  std::string device_names{};
#define REGISTER_CONVERTER(op_type_, func_name_, device_names_)   \
  device_names = #device_names_;                                  \
  device_names.erase(                                             \
      std::remove(device_names.begin(), device_names.end(), '"'), \
      device_names.end());                                        \
  device_names.erase(                                             \
      std::remove(device_names.begin(), device_names.end(), ' '), \
      device_names.end());                                        \
  for (auto& device_name : lite::Split(device_names, ",")) {      \
    target_supported_ops_[device_name].emplace(#op_type_);        \
    all_supported_ops_[#op_type_].emplace(device_name);           \
  }
#include "lite/kernels/nnadapter/converter/all.h"
#undef REGISTER_CONVERTER

// collect operators supported by mlu, bm
#define USE_SUBGRAPH_BRIDGE(op_type_, target_)        \
  target_supported_ops_[#target_].emplace(#op_type_); \
  all_supported_ops_[#op_type_].emplace(#target_);
#include "lite/kernels/bm/bridges/paddle_use_bridges.h"
#include "lite/kernels/mlu/bridges/paddle_use_bridges.h"
#undef USE_SUBGRAPH_BRIDGE
}

}  // namespace lite_api
}  // namespace paddle
