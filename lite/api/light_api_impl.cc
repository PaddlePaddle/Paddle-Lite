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
namespace lite {

void LightPredictorImpl::Init(const lite_api::MobileConfig& config) {
  // LightPredictor Only support NaiveBuffer backend in publish lib
  if (config.lite_model_file().empty()) {
    raw_predictor_.reset(
        new LightPredictor(config.model_dir(),
                           config.model_buffer(),
                           config.param_buffer(),
                           config.model_from_memory(),
                           lite_api::LiteModelType::kNaiveBuffer));
  } else {
    raw_predictor_.reset(new LightPredictor(config.lite_model_file(),
                                            config.model_from_memory()));
  }
  mode_ = config.power_mode();
  threads_ = config.threads();
}

std::unique_ptr<lite_api::Tensor> LightPredictorImpl::GetInput(int i) {
  return std::unique_ptr<lite_api::Tensor>(
      new lite_api::Tensor(raw_predictor_->GetInput(i)));
}

std::unique_ptr<const lite_api::Tensor> LightPredictorImpl::GetOutput(
    int i) const {
  return std::unique_ptr<lite_api::Tensor>(
      new lite_api::Tensor(raw_predictor_->GetOutput(i)));
}

void LightPredictorImpl::Run() {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Global().SetRunMode(mode_, threads_);
#endif
  raw_predictor_->Run();
}

std::shared_ptr<lite_api::PaddlePredictor> LightPredictorImpl::Clone() {
  LOG(FATAL) << "The Clone API is not supported in LigthPredictor";
}

std::string LightPredictorImpl::GetVersion() const { return lite::version(); }

std::unique_ptr<const lite_api::Tensor> LightPredictorImpl::GetTensor(
    const std::string& name) const {
  return std::unique_ptr<const lite_api::Tensor>(
      new lite_api::Tensor(raw_predictor_->GetTensor(name)));
}
std::unique_ptr<lite_api::Tensor> LightPredictorImpl::GetInputByName(
    const std::string& name) {
  return std::unique_ptr<lite_api::Tensor>(
      new lite_api::Tensor(raw_predictor_->GetInputByName(name)));
}

std::vector<std::string> LightPredictorImpl::GetInputNames() {
  return raw_predictor_->GetInputNames();
}

std::vector<std::string> LightPredictorImpl::GetOutputNames() {
  return raw_predictor_->GetOutputNames();
}

}  // namespace lite

namespace lite_api {

template <>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(
    const MobileConfig& config) {
  auto x = std::make_shared<lite::LightPredictorImpl>();
  x->Init(config);
  return x;
}

}  // namespace lite_api
}  // namespace paddle
