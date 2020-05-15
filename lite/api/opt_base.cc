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

#include "lite/api/opt_base.h"
#include "all_kernel_faked.cc"  // NOLINT

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
    LOG(FATAL) << "Unsupported Model type :" << optimize_out_type;
  }
}

void OptBase::SetPassesInternal(
    const std::vector<std::string>& passes_internal) {
  opt_config_.set_passes_internal(passes_internal);
}

void OptBase::SetValidPlaces(const std::string& valid_places) {
  valid_places_.clear();
  auto target_reprs = lite::Split(valid_places, ",");
  for (auto& target_repr : target_reprs) {
    if (target_repr == "arm") {
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
          Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)});
      valid_places_.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)});
      valid_places_.emplace_back(
          TARGET(kARM));  // enable kARM CPU kernel when no opencl kernel
    } else if (target_repr == "x86") {
      valid_places_.emplace_back(TARGET(kX86));
    } else if (target_repr == "npu") {
      valid_places_.emplace_back(TARGET(kNPU));
    } else if (target_repr == "xpu") {
      valid_places_.emplace_back(TARGET(kXPU));
    } else if (target_repr == "rknpu") {
      valid_places_.emplace_back(TARGET(kRKNPU));
      valid_places_.emplace_back(
          TARGET(kRKNPU), PRECISION(kInt8), DATALAYOUT(kNCHW));
    } else if (target_repr == "apu") {
      valid_places_.emplace_back(
          Place{TARGET(kAPU), PRECISION(kInt8), DATALAYOUT(kNCHW)});
    } else {
      LOG(FATAL) << lite::string_format(
          "Wrong target '%s' found, please check the command flag "
          "'valid_targets'",
          target_repr.c_str());
    }
  }
  CHECK(!valid_places_.empty())
      << "At least one target should be set, should set the "
         "command argument 'valid_targets'";
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
    auto resulted_model_name =
        record_strip_info_ ? "information of striped model" : "optimized model";
    std::cout << "Save the " << resulted_model_name
              << " into :" << lite_out_name_ << "successfully";
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
    auto resulted_model_name =
        record_strip_info_ ? "information of striped model" : "optimized model";
    std::cout << "Save the " << resulted_model_name
              << " into :" << lite_out_name_ << "successfully";
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
    LOG(FATAL) << "[" << model_set_dir_ << "] does not contain any model";
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

    std::cout << "Start optimize model: " << input_model_dir;

    opt_config_.set_model_dir(input_model_dir);
    opt_config_.set_model_file(model_file);
    opt_config_.set_param_file(param_file);

    auto opt_predictor = lite_api::CreatePaddlePredictor(opt_config_);
    opt_predictor->SaveOptimizedModel(
        lite_out_name_, model_type_, record_strip_info);

    std::cout << "Optimize done. ";
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
    std::cout << "Record the information of stripped models into :"
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
      "        `set_valid_places(arm|opencl|x86|npu|xpu|rknpu|apu)`\n"
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
      "------------------------------------------------------------------------"
      "-----------------------------------------------------------\n";
  std::cout << "opt version:" << opt_version << std::endl << help_info;
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
      "        `--valid_targets=(arm|opencl|x86|npu|xpu)`\n"
      "        `--record_tailoring_info=(true|false)`\n"
      "  Arguments of model checking and ops information:\n"
      "        `--print_all_ops=true`   Display all the valid operators of "
      "Paddle-Lite\n"
      "        `--print_supported_ops=true  "
      "--valid_targets=(arm|opencl|x86|npu|xpu)`"
      "  Display valid operators of input targets\n"
      "        `--print_model_ops=true  --model_dir=<model_param_dir> "
      "--valid_targets=(arm|opencl|x86|npu|xpu)`"
      "  Display operators in the input model\n";
  std::cout << "paddlelite opt version:" << opt_version << std::endl
            << help_info << std::endl;
}

// 2. Print supported info of inputed ops
void OptBase::PrintOpsInfo(const std::set<std::string>& valid_ops) {
  std::vector<std::string> lite_supported_targets = {"kHost",
                                                     "kX86",
                                                     "kCUDA",
                                                     "kARM",
                                                     "kOpenCL",
                                                     "kFPGA",
                                                     "kNPU",
                                                     "kXPU",
                                                     "kRKNPU",
                                                     "kAPU",
                                                     "kAny",
                                                     "kUnk"};
  // Get the lengh of the first column: maximum length of the op_type
  size_t maximum_optype_length = 0;
  for (auto it = supported_ops.begin(); it != supported_ops.end(); it++) {
    maximum_optype_length = it->first.size() > maximum_optype_length
                                ? it->first.size()
                                : maximum_optype_length;
  }
  std::cout << std::setiosflags(std::ios::internal);
  // Print the first row: OP_nam taget1 target2 ...
  std::cout << std::setw(maximum_optype_length) << "OP_name";
  for (size_t i = 0; i < lite_supported_targets.size(); i++) {
    std::cout << std::setw(10) << lite_supported_targets[i].substr(1);
  }
  std::cout << std::endl;
  // Print the name of supported ops and mark if it's supported by each target
  // print the support info of inputed ops: valid_ops
  for (auto op = valid_ops.begin(); op != valid_ops.end(); op++) {
    std::cout << std::setw(maximum_optype_length) << *op;
    // Check: If this kernel doesn't match any operator, we will skip it.
    if (supported_ops.find(*op) == supported_ops.end()) {
      continue;
    }
    // Print OP info.
    auto ops_valid_places = supported_ops.at(*op);
    for (size_t i = 0; i < lite_supported_targets.size(); i++) {
      if (std::find(ops_valid_places.begin(),
                    ops_valid_places.end(),
                    lite_supported_targets[i]) != ops_valid_places.end()) {
        std::cout << std::setw(10) << "Y";
      } else {
        std::cout << std::setw(10) << " ";
      }
    }
    std::cout << std::endl;
  }
}

void OptBase::DisplayKernelsInfo() {  // Display kernel information
  std::cout << ::paddle::lite::KernelRegistry::Global().DebugString();
}
void OptBase::PrintAllOps() {
  // 1. Get supported ops on these targets
  std::set<std::string> valid_ops;
  for (size_t i = 0; i < supported_ops_target.size(); i++) {
    auto ops = supported_ops_target[i];
    valid_ops.insert(ops.begin(), ops.end());
  }
  // 2. Print support info of these ops
  PrintOpsInfo(valid_ops);
}

void OptBase::PrintSupportedOps() {
  // 1. Get the valid hardware targets
  std::vector<TargetType> target_types = {};
  for (size_t i = 0; i < valid_places_.size(); i++) {
    target_types.push_back(valid_places_[i].target);
  }
  std::string targets_str = TargetToStr(target_types[0]);
  for (size_t i = 1; i < target_types.size(); i++) {
    targets_str = targets_str + TargetToStr(target_types[i]);
  }
  std::cout << "Supported OPs on '" << targets_str << "': " << std::endl;
  target_types.push_back(TARGET(kHost));
  target_types.push_back(TARGET(kUnk));

  // 2. Get supported ops on these targets
  std::set<std::string> valid_ops;
  for (size_t i = 0; i < target_types.size(); i++) {
    auto ops = supported_ops_target[static_cast<int>(target_types[i])];
    valid_ops.insert(ops.begin(), ops.end());
  }
  // 3. Print support info of these ops
  PrintOpsInfo(valid_ops);
}

// test whether this model is supported
void OptBase::CheckIfModelSupported(bool print_ops_info) {
  // 1. parse valid places and valid targets
  auto valid_ops = supported_ops_target[static_cast<int>(TARGET(kHost))];
  auto valid_unktype_ops = supported_ops_target[static_cast<int>(TARGET(kUnk))];
  valid_ops.insert(
      valid_ops.end(), valid_unktype_ops.begin(), valid_unktype_ops.end());
  for (size_t i = 0; i < valid_places_.size(); i++) {
    auto target = valid_places_[i].target;
    auto ops = supported_ops_target[static_cast<int>(target)];
    valid_ops.insert(valid_ops.end(), ops.begin(), ops.end());
  }
  // get valid ops
  std::set<std::string> valid_ops_set(valid_ops.begin(), valid_ops.end());

  // 2.Load model into program to get ops in model
  std::string prog_path = opt_config_.model_dir() + "/__model__";
  if (!(opt_config_.model_file()).empty() &&
      !(opt_config_.param_file()).empty()) {
    prog_path = opt_config_.model_file();
  }
  lite::cpp::ProgramDesc cpp_prog;
  framework::proto::ProgramDesc pb_proto_prog =
      *lite::LoadProgram(prog_path, false);
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
      if (valid_ops_set.count(op_type) == 0) {
        unsupported_ops.insert(op_type);
      }
    }
  }
  // 3. Print ops_info of input model and check if this model is supported
  if (print_ops_info) {
    std::cout << "OPs in the input model include:\n";
    PrintOpsInfo(input_model_ops);
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
    std::sort(targets.begin(), targets.end());
    targets.erase(unique(targets.begin(), targets.end()), targets.end());
    std::string targets_str = TargetToStr(targets[0]);
    for (size_t i = 1; i < targets.size(); i++) {
      targets_str = targets_str + "," + TargetToStr(targets[i]);
    }

    LOG(ERROR) << "Error: This model is not supported, because "
               << unsupported_ops.size() << " ops are not supported on '"
               << targets_str << "'. These unsupported ops are: '"
               << unsupported_ops_str << "'.";
    exit(1);
  }
  if (print_ops_info) {
    std::cout << "Paddle-Lite supports this model!" << std::endl;
    exit(1);
  }
}
}  // namespace lite_api
}  // namespace paddle
