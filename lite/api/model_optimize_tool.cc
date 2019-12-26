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
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#endif
// "supported_kernel_op_info.h", "all_kernel_faked.cc" and "kernel_src_map.h"
// are created automatically during model_optimize_tool's compiling period
#include "all_kernel_faked.cc"  // NOLINT
#include "kernel_src_map.h"     // NOLINT
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/op_registry.h"
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"
#include "supported_kernel_op_info.h"  // NOLINT

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
    "protobuf",
    "store type of the output optimized model. protobuf/naive_buffer");
DEFINE_bool(display_kernels, false, "Display kernel information");
DEFINE_bool(record_tailoring_info,
            false,
            "Record kernels and operators information of the optimized model "
            "for tailoring compiling, information are stored into optimized "
            "model path as hidden files");
DEFINE_string(optimize_out, "", "path of the output optimized model");
DEFINE_string(valid_targets,
              "arm",
              "The targets this model optimized for, should be one of (arm, "
              "opencl, x86), splitted by space");
DEFINE_bool(prefer_int8_kernel, false, "Prefer to run model with int8 kernels");
DEFINE_bool(print_supported_ops,
            false,
            "Print supported operators on the inputed target");
DEFINE_bool(print_all_ops,
            false,
            "Print all the valid operators of Paddle-Lite");
DEFINE_bool(print_model_ops,
            false,
            "Print all the valid operators of Paddle-Lite");
namespace paddle {
namespace lite_api {
//! Display the kernel information.
void DisplayKernels() {
  LOG(INFO) << ::paddle::lite::KernelRegistry::Global().DebugString();
}

std::vector<Place> ParserValidPlaces() {
  std::vector<Place> valid_places;
  auto target_reprs = lite::Split(FLAGS_valid_targets, ",");
  for (auto& target_repr : target_reprs) {
    if (target_repr == "arm") {
      valid_places.emplace_back(TARGET(kARM));
    } else if (target_repr == "opencl") {
      valid_places.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)});
      valid_places.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNHWC)});
      valid_places.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)});
      valid_places.emplace_back(
          Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)});
      valid_places.emplace_back(
          TARGET(kARM));  // enable kARM CPU kernel when no opencl kernel
    } else if (target_repr == "x86") {
      valid_places.emplace_back(TARGET(kX86));
    } else if (target_repr == "npu") {
      valid_places.emplace_back(TARGET(kNPU));
    } else if (target_repr == "xpu") {
      valid_places.emplace_back(TARGET(kXPU));
    } else {
      LOG(FATAL) << lite::string_format(
          "Wrong target '%s' found, please check the command flag "
          "'valid_targets'",
          target_repr.c_str());
    }
  }

  CHECK(!valid_places.empty())
      << "At least one target should be set, should set the "
         "command argument 'valid_targets'";

  if (FLAGS_prefer_int8_kernel) {
    LOG(WARNING) << "Int8 mode is only support by ARM target";
    valid_places.insert(valid_places.begin(),
                        Place{TARGET(kARM), PRECISION(kInt8)});
  }
  return valid_places;
}

void RunOptimize(const std::string& model_dir,
                 const std::string& model_file,
                 const std::string& param_file,
                 const std::string& optimize_out,
                 const std::string& optimize_out_type,
                 const std::vector<Place>& valid_places,
                 bool record_tailoring_info) {
  if (!model_file.empty() && !param_file.empty()) {
    LOG(WARNING)
        << "Load combined-param model. Option model_dir will be ignored";
  }

  lite_api::CxxConfig config;
  config.set_model_dir(model_dir);
  config.set_model_file(model_file);
  config.set_param_file(param_file);
  config.set_valid_places(valid_places);
  auto predictor = lite_api::CreatePaddlePredictor(config);

  LiteModelType model_type;
  if (optimize_out_type == "protobuf") {
    model_type = LiteModelType::kProtobuf;
  } else if (optimize_out_type == "naive_buffer") {
    model_type = LiteModelType::kNaiveBuffer;
  } else {
    LOG(FATAL) << "Unsupported Model type :" << optimize_out_type;
  }

  OpKernelInfoCollector::Global().SetKernel2path(kernel2path_map);
  predictor->SaveOptimizedModel(
      optimize_out, model_type, record_tailoring_info);
  if (record_tailoring_info) {
    LOG(INFO) << "Record the information of tailored model into :"
              << optimize_out;
  }
}

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

// Parse Input command
void ParseInputCommand(char** argv) {
  if (FLAGS_print_all_ops) {
    std::cout << "All OPs supported by Paddle-Lite: " << supported_ops.size()
              << " ops in total." << std::endl;
    for (auto it = supported_ops.begin(); it != supported_ops.end(); it++) {
      std::cout << it->first << " : ";
      auto ops_valid_places = it->second;
      for (int i = 0; i < ops_valid_places.size(); i++) {
        std::cout << ops_valid_places[i].substr(1) << " ";
      }
      std::cout << std::endl;
    }
    exit(1);
  } else if (FLAGS_print_supported_ops) {
    auto valid_places = paddle::lite_api::ParserValidPlaces();
    // get valid_targets string
    std::vector<TargetType> targets = {};
    for (int i = 0; i < valid_places.size(); i++) {
      targets.push_back(valid_places[i].target);
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(unique(targets.begin(), targets.end()), targets.end());
    std::string targets_str = TargetToStr(targets[0]);
    for (int i = 1; i < targets.size(); i++) {
      targets_str = targets_str + TargetToStr(targets[i]);
    }
    std::cout << "Supported OPs on '" << targets_str << "': " << std::endl;
    targets.push_back(TARGET(kHost));
    targets.push_back(TARGET(kUnk));
    std::string supported_ops_str = "{";
    for (int i = 0; i < targets.size(); i++) {
      auto ops = supported_ops_target[static_cast<int>(targets[i])];
      for (int i = 0; i < ops.size(); i++) {
        supported_ops_str = supported_ops_str + ops[i] + ", ";
      }
    }
    supported_ops_str =
        supported_ops_str.substr(0, supported_ops_str.length() - 2) + "}";
    std::cout << supported_ops_str << std::endl;
    exit(1);
  }
}
// test whether this model is supported
void CheckIfModelSupported() {
  auto valid_places = paddle::lite_api::ParserValidPlaces();
  // set valid_ops
  auto valid_ops = supported_ops_target[static_cast<int>(TARGET(kHost))];
  auto valid_unktype_ops = supported_ops_target[static_cast<int>(TARGET(kUnk))];
  valid_ops.insert(
      valid_ops.end(), valid_unktype_ops.begin(), valid_unktype_ops.end());
  for (int i = 0; i < valid_places.size(); i++) {
    auto target = valid_places[i].target;
    auto ops = supported_ops_target[static_cast<int>(target)];
    valid_ops.insert(valid_ops.end(), ops.begin(), ops.end());
  }
  std::set<std::string> valid_ops_set(valid_ops.begin(), valid_ops.end());
  // Load model
  std::string prog_path = FLAGS_model_dir + "/__model__";
  if (!FLAGS_model_file.empty() && !FLAGS_param_file.empty()) {
    prog_path = FLAGS_model_file;
  }
  lite::cpp::ProgramDesc cpp_prog;
  framework::proto::ProgramDesc pb_proto_prog =
      *lite::LoadProgram(prog_path, false);
  lite::pb::ProgramDesc pb_prog(&pb_proto_prog);
  // Transform to cpp::ProgramDesc
  lite::TransformProgramDescAnyToCpp(pb_prog, &cpp_prog);
  std::vector<std::string> unsupported_ops;
  std::vector<std::string> input_model_ops;
  auto main_block = cpp_prog.GetBlock<lite::cpp::BlockDesc>(0);
  for (size_t i = 0; i < main_block->OpsSize(); ++i) {
    auto& op_desc = *main_block->GetOp<lite::cpp::OpDesc>(i);
    auto op_type = op_desc.Type();
    input_model_ops.push_back(op_type);
    if (valid_ops_set.count(op_type) == 0) {
      unsupported_ops.push_back(op_type);
    }
  }
  std::sort(input_model_ops.begin(), input_model_ops.end());
  input_model_ops.erase(unique(input_model_ops.begin(), input_model_ops.end()),
                        input_model_ops.end());
  std::sort(unsupported_ops.begin(), unsupported_ops.end());
  unsupported_ops.erase(unique(unsupported_ops.begin(), unsupported_ops.end()),
                        unsupported_ops.end());

  if (FLAGS_print_model_ops) {
    std::string input_model_ops_str = "OPs in the input model include:\n{";
    for (int i = 0; i < input_model_ops.size(); i++) {
      input_model_ops_str = input_model_ops_str + input_model_ops[i] + ", ";
    }
    input_model_ops_str.erase(input_model_ops_str.end() - 1);
    input_model_ops_str =
        input_model_ops_str.substr(0, input_model_ops_str.length() - 2) + "}\n";
    std::cout << input_model_ops_str;
  }
  if (!unsupported_ops.empty()) {
    std::string unsupported_ops_str = unsupported_ops[0];
    for (int i = 1; i < unsupported_ops.size(); i++) {
      unsupported_ops_str = unsupported_ops_str + ", " + unsupported_ops[i];
    }
    std::vector<TargetType> targets = {};
    for (int i = 0; i < valid_places.size(); i++) {
      targets.push_back(valid_places[i].target);
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(unique(targets.begin(), targets.end()), targets.end());
    std::string targets_str = TargetToStr(targets[0]);
    for (int i = 1; i < targets.size(); i++) {
      targets_str = targets_str + "," + TargetToStr(targets[i]);
    }

    LOG(ERROR) << "Error: This model is not supported, because "
               << unsupported_ops.size() << " ops are not supported on '"
               << targets_str << "'. These unsupported ops are: '"
               << unsupported_ops_str << "'.";
    exit(1);
  }
  if (FLAGS_print_model_ops) {
    std::cout << "Paddle-Lite supports this model!" << std::endl;
    exit(1);
  }
}
void Main() {
  if (FLAGS_display_kernels) {
    DisplayKernels();
    exit(0);
  }

  auto valid_places = ParserValidPlaces();
  if (FLAGS_model_set_dir == "") {
    RunOptimize(FLAGS_model_dir,
                FLAGS_model_file,
                FLAGS_param_file,
                FLAGS_optimize_out,
                FLAGS_optimize_out_type,
                valid_places,
                FLAGS_record_tailoring_info);
    return;
  }

  if (!FLAGS_record_tailoring_info) {
    LOG(WARNING) << "--model_set_dir option only be used with "
                    "--record_tailoring_info=true together";
    return;
  }

  auto model_dirs = lite::ListDir(FLAGS_model_set_dir, true);
  if (model_dirs.size() == 0) {
    LOG(FATAL) << "[" << FLAGS_model_set_dir << "] does not contain any model";
  }
  // Optimize models in FLAGS_model_set_dir
  for (const auto& name : model_dirs) {
    std::string input_model_dir =
        lite::Join<std::string>({FLAGS_model_set_dir, name}, "/");
    std::string output_model_dir =
        lite::Join<std::string>({FLAGS_optimize_out, name}, "/");

    std::string model_file = "";
    std::string param_file = "";

    if (FLAGS_model_filename != "" && FLAGS_param_filename != "") {
      model_file =
          lite::Join<std::string>({input_model_dir, FLAGS_model_filename}, "/");
      param_file =
          lite::Join<std::string>({input_model_dir, FLAGS_param_filename}, "/");
    }

    LOG(INFO) << "Start optimize model: " << input_model_dir;
    RunOptimize(input_model_dir,
                model_file,
                param_file,
                output_model_dir,
                FLAGS_optimize_out_type,
                valid_places,
                FLAGS_record_tailoring_info);
    LOG(INFO) << "Optimize done. ";
  }

  // Collect all models information
  CollectModelMetaInfo(
      FLAGS_optimize_out, model_dirs, lite::TAILORD_OPS_SOURCE_LIST_FILENAME);
  CollectModelMetaInfo(
      FLAGS_optimize_out, model_dirs, lite::TAILORD_OPS_LIST_NAME);
  CollectModelMetaInfo(FLAGS_optimize_out,
                       model_dirs,
                       lite::TAILORD_KERNELS_SOURCE_LIST_FILENAME);
  CollectModelMetaInfo(
      FLAGS_optimize_out, model_dirs, lite::TAILORD_KERNELS_LIST_NAME);
}

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  // at least one argument should be inputed
  const std::string help_info =
      "At least one argument should be inputed. Valid arguments are listed "
      "below:\n"
      "        `--print_all_ops`   Display all the valid operators of "
      "Paddle-Lite\n"
      "        `--print_supported_ops=true  --valid_targets=(arm|opencl|x86)`"
      "  Display valid operators of input targets\n"
      "        `--print_model_ops=true  --model_dir=<model_param_dir> "
      "--valid_targets=(arm|opencl|x86)`"
      "  Display operators in the input model\n"
      "Arguments of model optimization:\n"
      "        `--model_dir=<model_param_dir>`\n"
      "        `--model_file=<model_path>`\n"
      "        `--param_file=<param_path>`\n"
      "        `--optimize_out_type=(protobuf|naive_buffer)`\n"
      "        `--optimize_out=<output_optimize_model_dir>`\n"
      "        `--valid_targets=(arm|opencl|x86)`\n"
      "        `--prefer_int8_kernel=(true|false)`\n"
      "        `--record_tailoring_info=(true|false)`";
  if (argc < 2) {
    std::cerr << help_info << std::endl;
    exit(1);
  }
  google::ParseCommandLineFlags(&argc, &argv, false);
  paddle::lite_api::ParseInputCommand(argv);
  paddle::lite_api::CheckIfModelSupported();
  paddle::lite_api::Main();
  return 0;
}
