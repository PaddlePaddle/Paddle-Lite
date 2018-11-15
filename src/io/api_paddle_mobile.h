/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains the implementation of inference API with Anakin engine
 * embeded, this API can only support Anakin models.
 */

#pragma once

#include <vector>
#include "io/paddle_inference_api.h"

// from paddle_mobile
#include "common/enforce.h"
#include "common/types.h"
#include "io/paddle_mobile.h"

namespace paddle_mobile {

template <typename Dtype = CPU, Precision P = Precision::FP32>
class PaddleMobilePredictor : public PaddlePredictor {
 public:
  PaddleMobilePredictor() = delete;

  explicit PaddleMobilePredictor(const PaddleMobileConfig& config);

  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override;

  double CaculatePredictTime() override;

  ~PaddleMobilePredictor() override;

 private:
  std::unique_ptr<PaddleMobile<Dtype, P>> paddle_mobile_;
  bool Init(const PaddleMobileConfig& config);

  PaddleMobileConfig config_;
};

}  // namespace paddle_mobile
