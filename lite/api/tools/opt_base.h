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

/*
 * This file defines Opt and basic functions about model transformation.
 */

#ifndef PADDLE_LITE_OPT_H_  // NOLINT
#define PADDLE_LITE_OPT_H_
#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <vector>
// stores the map that records the source_file path of each kernel.
#include "kernel_src_map.h"  // NOLINT
#include "lite/api/cxx_api.h"
// version of Paddle-lite
#include "lite/core/version.h"
// model parser functions to pre-load model to verify if this model is supported
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/utils/string.h"
// recorded all the ops supported by paddle-lite
#include "supported_kernel_op_info.h"  // NOLINT

namespace paddle {
namespace lite_api {

/// The PaddlePredictor defines the basic interfaces for different kinds of
/// predictors.
class LITE_API OptBase {
 public:
  OptBase() { InitSupportedOpInfo(); }
  void SetModelSetDir(const std::string &model_set_path);
  void SetModelDir(const std::string &model_dir_path);
  void SetModelFile(const std::string &model_path);
  void SetParamFile(const std::string &param_path);
  void EnableFloat16() { enable_fp16_ = true; }
  void SetValidPlaces(const std::string &valid_places);
  void SetOptimizeOut(const std::string &lite_out_name);
  void RecordModelInfo(bool record_strip_info = true);
  void SetQuantModel(bool quant_model);
  void SetQuantType(const std::string &quant_type);
  void SetSparseModel(bool sparse_model);
  void SetSparseThreshold(const float sparse_threshold = 0.6f);
  void SetNNAdapterMixedPrecisionQuantizationConfigPath(
      const std::string &nnadapter_mixed_precision_quantization_config_path);
  // set optimized_model type
  void SetModelType(std::string model_type = "naive_buffer");
  // internal inference for developer, not recommanded.
  // choose methods of model optimizing.
  void SetPassesInternal(const std::vector<std::string> &passes_internal = {});
  // transform and save the optimized model
  void Run();
  void RunOptimize(const std::string &model_dir_path = "",
                   const std::string &model_path = "",
                   const std::string &param_path = "",
                   const std::string &model_type = "",
                   const std::string &valid_places = "",
                   const std::string &optimized_out_path = "");
  // fuctions of printing info
  // 1. help info
  // 1.1 Print help info for opt python api
  void PrintHelpInfo();
  std::string OptVersion() { return lite::version(); }
  // 1.2 Print help info for executable opt bin
  void PrintExecutableBinHelpInfo();
  // 2. PrintOpsInfo
  void PrintOpsInfo(const std::set<std::string> &valid_ops = {},
                    const std::vector<std::string> valid_targets =
                        {"kARM",
                         "kOpenCL",
                         "kMetal",
                         "kXPU",
                         "kHost",
                         "kX86",
                         "kBM",
                         "kIntelFPGA",
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
                         "kUnK"});  // print supported ops on target_types
  void PrintAllOps();               // print all ops
  void PrintSupportedOps();         // print ops supported on valid_places_
  void DisplayKernelsInfo();        // Display kernel information
  // 3. Check if this model is supported
  void CheckIfModelSupported(bool print_ops_info = true);
  void PrintAllSupportedOpsInMdformat();  // print all ops in markdown format to
                                          // modify doc
  std::vector<std::string> VisualizeOptimizedNBModel(
      const std::string &model_dir, const std::string &output_path);

 private:
  bool enable_fp16_{false};
  CxxConfig opt_config_;
  // valid places for the optimized_model
  std::vector<Place> valid_places_;
  // filename of the optimized_model
  std::string lite_out_name_;
  // type of the optimized_model, kNaiveBuffer default.
  LiteModelType model_type_{LiteModelType::kNaiveBuffer};
  // Dir path of a set of models, this should be combined with model
  std::string model_set_dir_;
  bool record_strip_info_{false};
  std::map<std::string, std::set<std::string>> target_supported_ops_{};
  std::map<std::string, std::set<std::string>> all_supported_ops_{};
  void RunOptimizeFromModelSet(bool record_strip_info = false);
  void InitSupportedOpInfo();
};

}  // namespace lite_api
}  // namespace paddle

#endif  // NOLINT
