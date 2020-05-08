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

#include "include/data_reader.h"
#include <limits>

using std::string;
using std::vector;

int FEATURE_NUM = 13;
float rate = 0.8;

int get_samples(string line, vector<float>* feature, float* label) {
  std::istringstream reader(line);
  std::vector<float> numbers;
  do {
    // read as many numbers as possible.
    for (float number; reader >> number;) {
      numbers.push_back(number);
    }
    // consume and discard token from stream.
    if (reader.fail()) {
      reader.clear();
      std::string token;
      reader >> token;
    }
  } while (!reader.eof());

  assert(numbers.size() == FEATURE_NUM + 1);
  for (int i = 0; i < FEATURE_NUM; i++) {
    feature->push_back(numbers[i]);
  }
  *label = numbers[FEATURE_NUM];
  return 0;
}

int normalize(const vector<vector<float>>& origin_features,
              vector<vector<float>>* features,
              float rate) {
  int inf = std::numeric_limits<int>::max();
  vector<float> min_vec(FEATURE_NUM, static_cast<float>(inf));
  vector<float> max_vec(FEATURE_NUM, -(static_cast<float>(inf)));
  vector<float> sum_vec(FEATURE_NUM, 0);
  vector<float> avg_vec(FEATURE_NUM, 0);

  for (int i = 0; i < origin_features.size(); i++) {
    for (int j = 0; j < FEATURE_NUM; j++) {
      min_vec[j] = min(min_vec[j], origin_features[i][j]);
      max_vec[j] = max(max_vec[j], origin_features[i][j]);
      sum_vec[j] += origin_features[i][j];
    }
  }

  for (int i = 0; i < FEATURE_NUM; i++) {
    avg_vec[i] = sum_vec[i] / origin_features.size();
  }

  for (int i = 0; i < origin_features.size() * rate - 1; i++) {
    vector<float> feat;
    for (int j = 0; j < FEATURE_NUM; j++) {
      feat.push_back((origin_features[i][j] - avg_vec[j]) /
                     (max_vec[j] - min_vec[j]));
    }
    features->push_back(feat);
  }
}

int read_samples(const string fname,
                 vector<vector<float>>* features,
                 vector<float>* labels) {
  fstream fin;
  fin.open(fname);
  if (!static_cast<bool>(fin)) {
    return 1;
  }
  vector<vector<float>> origin_features;
  vector<string> lines;
  string line;
  while (getline(fin, line)) {
    lines.push_back(line);
  }
  fin.close();

  for (int i = 0; i < lines.size(); i++) {
    vector<float> feat;
    float lbl = 0;
    get_samples(lines[i], &feat, &lbl);
    origin_features.push_back(feat);
    if (i < lines.size() * rate - 1) {
      labels->push_back(lbl);
    }
  }

  cout << "finish read fata" << endl;
  normalize(origin_features, features, rate);
  assert(features->size() == labels->size());
  return 0;
}
