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

#include <stdio.h>
#include <vector>

#include "lite/backends/fpga/KD/alignment.h"
#include "lite/backends/fpga/KD/layout.hpp"

namespace paddle {
namespace zynqmp {

static struct None none_;
static struct NCHW nchw_;
static struct NHWC nhwc_;
static struct NC nc_;
static struct NHW nhw_;
static struct N n_;

class Shape {
 public:
  std::function<int(Shape& s)> aligment_fuction = [](Shape& s) {  // NOLINT
    return s.layout_->alignedElementCount(s.dims_);
  };

  explicit Shape(std::vector<int> dims) { dims_ = dims; }

  Shape(LayoutType type, std::vector<int> dims) {
    dims_ = dims;
    setLayoutType(type);
  }

  Shape(const Shape& src) {
    dims_ = src.dims_;
    setLayoutType(src.layoutType_);
  }

  void setAligmentFunction(std::function<int(Shape& s)> f) {  // NOLINT
    aligment_fuction = f;
  }

  bool shouldAlign() {
    return layout_->alignedElementCount(dims_) != layout_->elementCount(dims_);
  }

  int num() {
    int index = layout_->numIndex();
    return index == -1 ? 1 : dims_[index];
  }

  int channel() {
    int index = layout_->channelIndex();
    return index == -1 ? 1 : dims_[index];
  }

  int height() {
    int index = layout_->heightIndex();
    return index == -1 ? 1 : dims_[index];
  }

  int width() {
    int index = layout_->widthIndex();
    return index == -1 ? 1 : dims_[index];
  }

  int dimSize() { return dims_.size(); }

  std::vector<int> dims() { return dims_; }

  size_t memorySize(int cellSize) { return aligment_fuction(*this) * cellSize; }

  int numel() { return layout_->elementCount(dims_); }

  int alignedElementCount() { return aligment_fuction(*this); }

  void setLayoutType(LayoutType layout) {
    this->layoutType_ = layout;
    switch (layout) {
      case None:
        layout_ = &none_;
        break;
      case NCHW:
        layout_ = &nchw_;
        break;
      case NHWC:
        layout_ = &nhwc_;
        break;
      case NC:
        layout_ = &nc_;
        break;
      case NHW:
        layout_ = &nhw_;
        break;
      case N:
        layout_ = &n_;
        break;
      default:
        break;
    }
  }

  void print() {}

  int operator[](int index) { return dims_[index]; }

 private:
  LayoutType layoutType_;
  Layout* layout_ = &nhwc_;
  std::vector<int> dims_;
};

}  // namespace zynqmp
}  // namespace paddle
