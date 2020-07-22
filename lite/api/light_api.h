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
 * This file implements a light-weight API which can run on mobile. We limit the
 * dependencies and the runtime computation complexity.
 */
#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/core/context.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"
#include "lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

/*
 * The light weight predictor, mainly for mobile. It loads an optimized model,
 * and will not depend on the MIR or perform latter optimization.
 */
class LITE_API LightPredictor {
 public:
  // constructor function of LightPredictor, `lite_model_file` refers to data in
  // model file or buffer,`model_from_memory` refers to whther to load model
  // from memory.
  LightPredictor(const std::string& lite_model_file,
                 bool model_from_memory = false) {
    scope_ = std::make_shared<Scope>();
    program_desc_ = std::make_shared<cpp::ProgramDesc>();
    Build(lite_model_file, model_from_memory);
  }

  // NOTE: This is a deprecated API and will be removed in latter release.
  LightPredictor(const std::string& model_dir,
                 const std::string& model_buffer = "",
                 const std::string& param_buffer = "",
                 bool model_from_memory = false,
                 lite_api::LiteModelType model_type =
                     lite_api::LiteModelType::kNaiveBuffer) {
    scope_ = std::make_shared<Scope>();
    program_desc_ = std::make_shared<cpp::ProgramDesc>();
    Build(model_dir, model_buffer, param_buffer, model_type, model_from_memory);
  }

  void Run() { program_->Run(); }

  // Get offset-th col of feed inputs.
  Tensor* GetInput(size_t offset);
  // get input by name.
  Tensor* GetInputByName(const std::string& name);
  // Get offset-th col of fetch outputs.
  const Tensor* GetOutput(size_t offset);

  const lite::Tensor* GetTensor(const std::string& name) const {
    auto* var = program_->exec_scope()->FindVar(name);
    return &var->Get<lite::Tensor>();
  }

  // get inputnames and get outputnames.
  std::vector<std::string> GetInputNames();
  std::vector<std::string> GetOutputNames();
  void PrepareFeedFetch();
  Scope* scope() { return scope_.get(); }

 private:
  void Build(const std::string& lite_model_file,
             bool model_from_memory = false);

  // NOTE: This is a deprecated API and will be removed in latter release.
  void Build(
      const std::string& model_dir,
      const std::string& model_buffer,
      const std::string& param_buffer,
      lite_api::LiteModelType model_type = lite_api::LiteModelType::kProtobuf,
      bool model_from_memory = false);

  void BuildRuntimeProgram(
      const std::shared_ptr<const cpp::ProgramDesc>& program_desc);

  void DequantizeWeight();

 private:
  std::shared_ptr<Scope> scope_;
  std::unique_ptr<RuntimeProgram> program_;
  std::shared_ptr<cpp::ProgramDesc> program_desc_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

class LightPredictorImpl : public lite_api::PaddlePredictor {
 public:
  LightPredictorImpl() = default;

  std::unique_ptr<lite_api::Tensor> GetInput(int i) override;

  std::unique_ptr<const lite_api::Tensor> GetOutput(int i) const override;

  void Run() override;

  std::shared_ptr<lite_api::PaddlePredictor> Clone() override;
  std::shared_ptr<lite_api::PaddlePredictor> Clone(
      const std::vector<std::string>& var_names) override;
  std::string GetVersion() const override;
  std::vector<std::string> GetInputNames() override;
  std::vector<std::string> GetOutputNames() override;

  std::unique_ptr<const lite_api::Tensor> GetTensor(
      const std::string& name) const override;
  // Get InputTebsor by name
  std::unique_ptr<lite_api::Tensor> GetInputByName(
      const std::string& name) override;

  void Init(const lite_api::MobileConfig& config);

 private:
  std::unique_ptr<lite::LightPredictor> raw_predictor_;
};

}  // namespace lite
}  // namespace paddle
