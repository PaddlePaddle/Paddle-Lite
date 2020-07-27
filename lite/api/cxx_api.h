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
#include <mutex>  //NOLINT
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

static const char TAILORD_OPS_SOURCE_LIST_FILENAME[] =
    ".tailored_ops_source_list";
static const char TAILORD_OPS_LIST_NAME[] = ".tailored_ops_list";
static const char TAILORD_KERNELS_SOURCE_LIST_FILENAME[] =
    ".tailored_kernels_source_list";
static const char TAILORD_KERNELS_LIST_NAME[] = ".tailored_kernels_list";

std::vector<std::string> GetAllOps();

/*
 * Predictor for inference, input a model, it will optimize and execute it.
 */
class LITE_API Predictor {
 public:
  // Create an empty predictor.
  Predictor() {
    scope_ = std::make_shared<Scope>();
    program_desc_ = std::make_shared<cpp::ProgramDesc>();
  }

  ///////////////////////////////////////////////////////////////////
  // Function: Predictor
  // Usage: Constructor of Predictor. Create a predictor with the
  // weight variable scope set given.
  ///////////////////////////////////////////////////////////////////
  explicit Predictor(const std::shared_ptr<lite::Scope>& root_scope)
      : scope_(root_scope) {}
  ///////////////////////////////////////////////////////////////////
  // Function: Predictor
  // Usage: Constructor of Predictor. This constructor function can
  // only be called in Predictor->Clone. This Function will create
  // a predictor from existed ProgramDesc, Scope and RuntimeProgram.
  ///////////////////////////////////////////////////////////////////
  Predictor(const std::shared_ptr<cpp::ProgramDesc>& program_desc,
            const std::shared_ptr<Scope>& root,
            const std::vector<Place>& valid_places,
            const std::vector<std::string>& var_names = {})
      : program_desc_(program_desc), scope_(root) {
    // step1. Create a Program to construct the exec_scope and ops
    Program program(program_desc_, scope_, valid_places, var_names);
    exec_scope_ = program.exec_scope();
    valid_places_ = valid_places;

    // step3. Create the RuntimeProgram.
    program_.reset(
        new RuntimeProgram(program_desc_, exec_scope_, kRootBlockIdx));
    program_generated_ = true;
  }

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

  void Build(const std::shared_ptr<cpp::ProgramDesc>& program_desc,
             const std::vector<Place>& valid_places,
             const std::vector<std::string>& passes = {});

  //////////////////////////////////////////////////////////
  // Function: Clone
  // Usage: Create a Predictor from an existed one,
  // the cloned predictor will share persistable variables
  // in scope_ with the original predictor.
  //////////////////////////////////////////////////////////
  std::shared_ptr<Predictor> Clone() {
    // step 1. Generate runtime_program, update op_info and var_info in
    // program_desc_
    if (!program_generated_) {
      GenRuntimeProgram();
    }
    program_->SaveToProgram(program_desc_);
    // step 2. Create a predictor friom current program_desc_ and
    // runtime_program.
    auto predictor =
        std::make_shared<Predictor>(program_desc_, scope_, valid_places_);
    // step3. Return the result
    return predictor;
  }
  //////////////////////////////////////////////////////////
  // Function: Clone(var_names)
  // Usage: Create a Predictor from an existed one,
  // the cloned predictor will share persistable variables
  // but persistable variables of name var_names will not
  // be shared.
  //////////////////////////////////////////////////////////
  std::shared_ptr<Predictor> Clone(const std::vector<std::string>& var_names) {
    CHECK(program_desc_) << "Both program and scope of current predicotr "
                            "should be not be nullptr in Clone mode.";
    CHECK(scope_) << "Both program and scope of current predicotr should be "
                     "not be nullptr in Clone mode.";
    // step 1. Generate runtime_program, update op_info and var_info in
    // program_desc_
    if (!program_generated_) {
      GenRuntimeProgram();
    }
    program_->SaveToProgram(program_desc_);
    // step 2. Create a predictor friom current program_desc_ and
    // runtime_program.
    auto predictor = std::make_shared<Predictor>(
        program_desc_, scope_, valid_places_, var_names);
    // step3. Copy some persistable variables into private scope.
    for (auto var_name : var_names) {
      predictor->exec_scope_->LocalVar(var_name);
      auto* tensor =
          predictor->scope_->Var(var_name)->GetMutable<lite::Tensor>();
      auto* sub_tensor =
          predictor->exec_scope_->Var(var_name)->GetMutable<Tensor>();
      sub_tensor->CopyDataFrom(*tensor);
    }
    // step4. Return the result
    return predictor;
  }

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
  // get param names
  std::vector<std::string> GetParamNames();

  void PrepareFeedFetch();

  // Get offset-th col of fetch results.
  const lite::Tensor* GetOutput(size_t offset) const;
  std::vector<const lite::Tensor*> GetOutputs() const;

  const cpp::ProgramDesc& program_desc() const;
  // get a mutable tensor according to its name
  lite::Tensor* GetMutableTensor(const std::string& name);
  // get a const tensor according to its name
  const lite::Tensor* GetTensor(const std::string& name) const;
  const RuntimeProgram& runtime_program() const;
  Scope* scope() { return scope_.get(); }

  // This method is disabled in mobile, for unnecessary dependencies required.
  void SaveModel(
      const std::string& dir,
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool record_info = false);
  void SaveOpKernelInfo(const std::string& model_dir);

  // #ifdef LITE_WITH_TRAIN
  //   void Run(const std::vector<framework::Tensor>& tensors) {
  //     FeedVars(tensors);
  //     program_->Run();
  //   }

  //   void FeedVars(const std::vector<framework::Tensor>& tensors);
  // #endif

 private:
  Optimizer optimizer_;
  std::shared_ptr<cpp::ProgramDesc> program_desc_;
  std::shared_ptr<Scope> scope_;
  Scope* exec_scope_;
  std::shared_ptr<RuntimeProgram> program_;
  bool program_generated_{false};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<Place> valid_places_;
};

class CxxPaddleApiImpl : public lite_api::PaddlePredictor {
 public:
  CxxPaddleApiImpl() {
    raw_predictor_ = std::make_shared<Predictor>();
    status_is_cloned_ = false;
  }
  explicit CxxPaddleApiImpl(const std::shared_ptr<Predictor>& raw_predictor)
      : raw_predictor_(raw_predictor) {
    status_is_cloned_ = true;
  }

  /// Create a new predictor from a config.
  void Init(const lite_api::CxxConfig& config);

  std::unique_ptr<lite_api::Tensor> GetInput(int i) override;

  std::unique_ptr<const lite_api::Tensor> GetOutput(int i) const override;

  void Run() override;

  std::shared_ptr<lite_api::PaddlePredictor> Clone() override;

  std::shared_ptr<lite_api::PaddlePredictor> Clone(
      const std::vector<std::string>& var_names) override;

  std::string GetVersion() const override;

  // get inputs names and get outputs names
  std::vector<std::string> GetInputNames() override;
  std::vector<std::string> GetOutputNames() override;
  // get param names
  std::vector<std::string> GetParamNames() override;

  // get tensor according to tensor's name
  std::unique_ptr<const lite_api::Tensor> GetTensor(
      const std::string& name) const override;
  // get a mutable tensor according to tensor's name
  std::unique_ptr<lite_api::Tensor> GetMutableTensor(
      const std::string& name) override;

  // Get InputTebsor by name
  std::unique_ptr<lite_api::Tensor> GetInputByName(
      const std::string& name) override;

  void SaveOptimizedModel(
      const std::string& model_dir,
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool record_info = false) override;

 private:
  std::shared_ptr<Predictor> raw_predictor_;
  lite_api::CxxConfig config_;
  std::mutex mutex_;
  bool status_is_cloned_;
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
