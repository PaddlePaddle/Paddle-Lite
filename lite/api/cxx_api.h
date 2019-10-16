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

#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/core/op_lite.h"
#include "lite/core/optimizer.h"
#include "lite/core/program.h"
#include "lite/core/types.h"
#include "lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

/*
 * Predictor for inference, input a model, it will optimize and execute it.
 */
class LITE_API Predictor {
 public:
  // Create an empty predictor.
  Predictor() { scope_ = std::make_shared<Scope>(); }
  // Create a predictor with the weight variable scope set.
  explicit Predictor(const std::shared_ptr<lite::Scope>& root_scope)
      : scope_(root_scope) {}

  // Build from a model, with places set for hardware config.
  void Build(
      const lite_api::CxxConfig& config,
      const std::vector<Place>& valid_places,
      const std::vector<std::string>& passes = {},
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf);

  void Build(
      const std::string& model_path,
      const std::string& model_file_path,
      const std::string& param_file_path,
      const std::vector<Place>& valid_places,
      const std::vector<std::string>& passes = {},
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool memory_from_memory = false);

  void Build(const cpp::ProgramDesc& desc,
             const std::vector<Place>& valid_places,
             const std::vector<std::string>& passes = {});

  void GenRuntimeProgram();

  // Run the predictor for a single batch of data.
  void Run() {
    if (!program_generated_) {
      GenRuntimeProgram();
    }
    program_->Run();
  }

  // Get offset-th col of feed inputs.
  lite::Tensor* GetInput(size_t offset);
  // get input by name.
  lite::Tensor* GetInputByName(const std::string& name);
  // get inputnames and get outputnames.
  std::vector<std::string> GetInputNames();
  std::vector<std::string> GetOutputNames();
  void PrepareFeedFetch();

  // Get offset-th col of fetch results.
  const lite::Tensor* GetOutput(size_t offset) const;
  const std::vector<lite::Tensor>* GetOutputs() const;

  const cpp::ProgramDesc& program_desc() const;
  const lite::Tensor* GetTensor(const std::string& name) const;
  const RuntimeProgram& runtime_program() const;

  // This method is disabled in mobile, for unnecessary dependencies required.
  void SaveModel(
      const std::string& dir,
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf);

#ifdef LITE_WITH_TRAIN
  void Run(const std::vector<framework::Tensor>& tensors) {
    FeedVars(tensors);
    program_->Run();
  }

  void FeedVars(const std::vector<framework::Tensor>& tensors);
#endif

 private:
  Optimizer optimizer_;
  cpp::ProgramDesc program_desc_;
  std::shared_ptr<Scope> scope_;
  const Scope* exec_scope_;
  std::unique_ptr<RuntimeProgram> program_;
  bool program_generated_{false};
  std::map<size_t, std::string> input_names_;
  std::map<std::string, size_t> idx2feeds_;
  std::map<size_t, std::string> output_names_;
};

/*
 * An executor for training.
 *
 * Usage:
 *
 * CXXTrainer trainer(...);
 * trainer.RunStartupProgram(...);
 * auto exe = BuildMainProgramExecutor(...);
 *
 * for (auto& epoch : epoches) {
 *   auto* tensor0 = exe.GetInput(...);
 *   // fill data for tensor0
 *   exe.Run();
 * }
#ifdef LITE_WITH_X86
class LITE_API CXXTrainer {
 public:
  CXXTrainer(const std::shared_ptr<lite::Scope>& root_scope,
             const std::vector<Place>& valid_places)
      : scope_(root_scope),
        valid_places_(valid_places),
        main_program_executor_(Predictor(scope_)) {}

  // Build the RuntimeProgram cache for the main program. The cache will run
  // multiple times for the epoches.
  // NOTE Just support to execute the 0-th block currently.
  Predictor& BuildMainProgramExecutor(const framework::proto::ProgramDesc& desc,
                                      int block_id = 0) {
    main_program_executor_.Build(desc, valid_places_);
    return main_program_executor_;
  }

#ifdef LITE_WITH_TRAIN
  Predictor& BuildMainProgramExecutor(framework::ProgramDesc& desc) {  // NOLINT
    return BuildMainProgramExecutor(*desc.Proto());
  }

  void RunStartupProgram(framework::ProgramDesc& desc) {  // NOLINT
    RunStartupProgram(*desc.Proto());
  }
#endif

  // Run the startup program. It just executes once, no cache needed.
  void RunStartupProgram(const framework::proto::ProgramDesc& desc,
                         int block_id = 0) {
    Predictor exe(scope_);
    exe.Build(desc,  valid_places_);
    exe.Run();
  }

 private:
  std::shared_ptr<lite::Scope> scope_;
  std::vector<Place> valid_places_;

  // The training program.
  Predictor main_program_executor_;
};
#endif
*/

}  // namespace lite
}  // namespace paddle
