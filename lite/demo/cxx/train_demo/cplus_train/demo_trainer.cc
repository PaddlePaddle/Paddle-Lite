// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "include/data_reader.h"
#include "paddle_api.h"  // NOLINT

using namespace paddle::lite_api;  // NOLINT

class LRModel {
 public:
  void InitModel() {
    // 1. Set CxxConfig
    CxxConfig config;
    config.set_model_dir("model_dir");
    std::vector<Place> valid_places{Place{TARGET(kARM), PRECISION(kFloat)}};
    config.set_valid_places(valid_places);
    predictor_ = CreatePaddlePredictor<CxxConfig>(config);
  }

  float Predict(const vector<vector<float>>& features,
                const vector<float>& labels) {
    // Create Tensor
    assert(features.size() == labels.size());
    int batch_size = features.size();
    std::unique_ptr<Tensor> input_tensor(std::move(predictor_->GetInput(0)));
    input_tensor->Resize(shape_t({batch_size, FEATURE_NUM}));
    auto* data = input_tensor->mutable_data<float>();
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < FEATURE_NUM; j++) {
        data[FEATURE_NUM * i + j] = features[i][j];
      }
    }
    std::unique_ptr<Tensor> y_tensor(std::move(predictor_->GetInput(1)));
    y_tensor->Resize(shape_t({batch_size, 1}));
    auto* y_data = y_tensor->mutable_data<float>();
    for (int i = 0; i < batch_size; i++) {
      y_data[i] = labels[i];
    }
    predictor_->Run();
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor_->GetOutput(0)));
    return output_tensor->data<float>()[0];
  }

 private:
  std::shared_ptr<PaddlePredictor> predictor_;
};

int shuffle(vector<vector<float>>* features, vector<float>* labels) {
  assert(features->size() == labels->size());
  vector<int> index;
  for (int i = 0; i < features->size(); i++) {
    index.push_back(i);
  }
  random_shuffle(index.begin(), index.end());

  vector<vector<float>> tmp_features;
  vector<float> tmp_labels;

  for (int i = 0; i < features->size(); i++) {
    tmp_features.push_back((*features)[index[i]]);
    tmp_labels.push_back((*labels)[index[i]]);
  }

  for (int i = 0; i < features->size(); i++) {
    for (int j = 0; j < FEATURE_NUM; j++) {
      (*features)[i][j] = tmp_features[i][j];
    }
    (*labels)[i] = tmp_labels[i];
  }
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "usage: ./demo_trainer is_small" << endl;
    cerr << "       if is_small is true, the batch size is set to 1, " << endl;
    cerr << "       and it will only runs for 10 steps." << endl;
    return 1;
  }
  string is_small = argv[1];
  vector<vector<float>> features;
  vector<float> labels;
  read_samples("housing.data", &features, &labels);
  cout << "sample count: " << features.size() << " " << endl;

  std::shared_ptr<LRModel> local_model(new LRModel());
  local_model->InitModel();

  if (is_small == "true") {
    cout << "small mode" << endl;
    for (int i; i < 10; i++) {
      vector<vector<float>> batch_feature;
      vector<float> batch_label;
      batch_feature.push_back(features[i]);
      batch_label.push_back(labels[i]);
      auto loss = local_model->Predict(batch_feature, batch_label);
      cout << "sample " << i << ": " << loss << endl;
    }
  } else if (is_small == "false") {
    // shuffle
    cout << "full model" << endl;
    int epoch = 100;
    int batch_size = 20;
    int step = 0;
    for (int i; i < epoch; i++) {
      shuffle(&features, &labels);
      for (int j = 0;
           j < ceil(static_cast<float>(features.size()) / batch_size);
           j++) {
        int start_idx = j * batch_size;
        int end_idx =
            min((j + 1) * batch_size, static_cast<int>(features.size()));
        auto batch_feature = vector<vector<float>>(features.begin() + start_idx,
                                                   features.begin() + end_idx);
        auto batch_label =
            vector<float>(labels.begin() + start_idx, labels.begin() + end_idx);
        auto loss = local_model->Predict(batch_feature, batch_label);
        if (step % 10 == 0) {
          std::cout << "batch: " << i << ", step: " << step
                    << ", Loss: " << loss << endl;
        }
        step += 1;
      }
    }
  } else {
    cerr << "wrong arg for is_small: " << is_small << endl;
  }
}
