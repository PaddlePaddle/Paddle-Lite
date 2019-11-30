/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "preprocess_conf.hpp"
#include <vector>


extern bool use_preprocess = false;

extern std::vector<float> preprocess_mean = {0, 0, 0};

extern std::vector<float> preprocess_scale = {0, 0, 0}; 

extern bool use_yolov3_416 = false;

extern float img_shape_width = 0;

extern float img_shape_height = 0;

