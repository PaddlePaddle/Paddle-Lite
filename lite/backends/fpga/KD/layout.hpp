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

#pragma once

#include <vector>

#include "lite/backends/fpga/KD/alignment.h"

namespace paddle {
namespace zynqmp {

enum LayoutType {
  None,
  N,
  NC,
  NCHW,
  CNHW,
  NHWC,
  NHW,
};

class Layout {
 public:
  virtual int numIndex() = 0;
  virtual int channelIndex() { return -1; }
  virtual int heightIndex() { return -1; }
  virtual int widthIndex() { return -1; }
  virtual int alignedElementCount(const std::vector<int>& dims) = 0;
  virtual int elementCount(const std::vector<int>& dims) = 0;
};

struct None : Layout {
  int numIndex() { return -1; }
  int channelIndex() { return -1; }
  int heightIndex() { return -1; }
  int widthIndex() { return -1; }
  int alignedElementCount(const std::vector<int>& dims) { return 16; }
  virtual int elementCount(const std::vector<int>& dims) { return 1; }
};

struct NCHW : Layout {
  int numIndex() { return 0; }
  int channelIndex() { return 1; }
  int heightIndex() { return 2; }
  int widthIndex() { return 3; }
  int alignedElementCount(const std::vector<int>& dims) {
    return dims[0] * dims[2] * align_image(dims[1] * dims[3]);
  }
  virtual int elementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1] * dims[2] * dims[3];
  }
};

struct NHWC : Layout {
  int numIndex() { return 0; }
  int heightIndex() { return 1; }
  int widthIndex() { return 2; }
  int channelIndex() { return 3; }
  int alignedElementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1] * align_image(dims[2] * dims[3]);
  }
  virtual int elementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1] * dims[2] * dims[3];
  }
};

struct CNHW : Layout {
  int numIndex() { return 1; }
  int channelIndex() { return 0; }
  int heightIndex() { return 2; }
  int widthIndex() { return 3; }
  int alignedElementCount(const std::vector<int>& dims) {
    return dims[1] * dims[2] * align_image(dims[0] * dims[3]);
  }
  int elementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1] * dims[2] * dims[3];
  }
};

struct NC : Layout {
  int numIndex() { return 0; }
  int channelIndex() { return 1; }
  int alignedElementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1];
  }
  virtual int elementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1];
  }
};

struct N : Layout {
  int numIndex() { return 0; }
  int alignedElementCount(const std::vector<int>& dims) { return dims[0]; }
  virtual int elementCount(const std::vector<int>& dims) { return dims[0]; }
};

struct NHW : Layout {
  int numIndex() { return 0; }
  int heightIndex() { return 1; }
  int widthIndex() { return 2; }
  int alignedElementCount(const std::vector<int>& dims) {
    // TODO(chonwhite) align it;
    return dims[0] * dims[1] * dims[2];
  }
  virtual int elementCount(const std::vector<int>& dims) {
    return dims[0] * dims[1] * dims[2];
  }
};

}  // namespace zynqmp
}  // namespace paddle
