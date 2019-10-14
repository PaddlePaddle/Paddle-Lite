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

#include "lite/api/light_api.h"
#include <string>
#include "lite/api/paddle_api.h"
#include "lite/core/version.h"
#include "lite/model_parser/model_parser.h"

namespace paddle {
namespace lite_api {

class LightPredictorImpl : public PaddlePredictor {
 public:
  LightPredictorImpl() = default;

  std::unique_ptr<Tensor> GetInput(int i) override;

  std::unique_ptr<const Tensor> GetOutput(int i) const override;

  void Run() override;

  std::string GetVersion() const override;
  std::vector<std::string> GetInputNames() override;
  std::vector<std::string> GetOutputNames() override;

  std::unique_ptr<const Tensor> GetTensor(
      const std::string& name) const override;
  // Get InputTebsor by name
  std::unique_ptr<Tensor> GetInputByName(const std::string& name) override;

  void Init(const MobileConfig& config);

 private:
  std::unique_ptr<lite::LightPredictor> raw_predictor_;
};

void LightPredictorImpl::Init(const MobileConfig& config) {
  // LightPredictor Only support NaiveBuffer backend in publish lib
  raw_predictor_.reset(new lite::LightPredictor(config.model_dir(),
                                                config.model_buffer(),
                                                config.param_buffer(),
                                                config.model_from_memory(),
                                                LiteModelType::kNaiveBuffer));
  raw_predictor_->PrepareFeedFetch();
}

std::unique_ptr<Tensor> LightPredictorImpl::GetInput(int i) {
  return std::unique_ptr<Tensor>(new Tensor(raw_predictor_->GetInput(i)));
}

std::unique_ptr<const Tensor> LightPredictorImpl::GetOutput(int i) const {
  return std::unique_ptr<Tensor>(new Tensor(raw_predictor_->GetOutput(i)));
}

void LightPredictorImpl::Run() { raw_predictor_->Run(); }

std::string LightPredictorImpl::GetVersion() const { return lite::version(); }

std::unique_ptr<const Tensor> LightPredictorImpl::GetTensor(
    const std::string& name) const {
  return std::unique_ptr<const Tensor>(
      new Tensor(raw_predictor_->GetTensor(name)));
}
std::unique_ptr<Tensor> LightPredictorImpl::GetInputByName(
    const std::string& name) {
  return std::unique_ptr<Tensor>(
      new Tensor(raw_predictor_->GetInputByName(name)));
}

std::vector<std::string> LightPredictorImpl::GetInputNames() {
  return raw_predictor_->GetInputNames();
}

std::vector<std::string> LightPredictorImpl::GetOutputNames() {
  return raw_predictor_->GetOutputNames();
}

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const MobileConfig& config) {
  auto x = std::make_shared<LightPredictorImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
}  // namespace paddle
