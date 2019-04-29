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
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "float16.hpp"
#include "llapi/zynqmp_api.h"
#include "shape.hpp"

namespace paddle_mobile {
namespace zynqmp {

enum DataType : int {
  FP32 = 0,
  FP16 = 1,
  INT8 = 2,
};

typedef uint16_t float16;

inline int CellSize(DataType type) {
  switch (type) {
    case FP32:
      return sizeof(float);
    case FP16:
      return sizeof(float16);
    case INT8:
      return sizeof(int8_t);
    default:
      return 0;
  }
  return 0;
}

class PlaceHolder {
 public:
  explicit PlaceHolder(size_t size) {
    size_ = size;
    data_ = fpga_malloc(size_);
  }

  void* data() { return data_; }

  size_t memorySize() { return size_; }

  ~PlaceHolder() {
    //    std::cout << "place holder dealloc";
    fpga_free(data_);
  }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

class Tensor {
 public:
  int id() { return id_; }

  template <typename Dtype>
  Dtype* data() {
    if (placeHolder_ == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<Dtype*>(this->placeHolder_->data());
  }

  template <typename Dtype>
  Dtype* mutableData(DataType dataType, const Shape& shape) {
    // if (this->shape_ != &shape) {
    if (this->shape_ != nullptr) {
      delete shape_;
    }
    this->shape_ = new Shape(shape);
    // }
    this->dataType_ = dataType;
    return mutableData<Dtype>();
  }

  template <typename Dtype>
  Dtype* mutableData() {
    size_t memorySize = shape_->memorySize(CellSize(dataType_));
    if (placeHolder_ != nullptr) {
      if (memorySize > placeHolder_->memorySize()) {
        delete placeHolder_;
        placeHolder_ = new PlaceHolder(memorySize);
      }
    } else {
      placeHolder_ = new PlaceHolder(memorySize);
    }
    return reinterpret_cast<Dtype*>(placeHolder_->data());
  }

  size_t memorySize() {
    size_t memorySize = shape_->memorySize(CellSize(dataType_));
    return memorySize;
  }

  void setDataType(DataType dataType) { this->dataType_ = dataType; }

  DataType dataType() { return this->dataType_; }

  Shape& shape() { return *shape_; }

  bool aligned() { return this->aligned_; }

  void setAligned(bool aligned) { this->aligned_ = aligned; }

  float* scale() { return scale_; }

  void alignImage(Tensor* dst = nullptr, bool copy = false) {
    if (shape_->shouldAlign()) {
      int cell_size = CellSize(this->dataType_);
      char* dst_data = nullptr;
      size_t mem_size = shape_->memorySize(cell_size);
      if (dst == nullptr) {
        dst_data = reinterpret_cast<char*>(fpga_malloc(mem_size));
      } else {
        dst_data = dst->data<char>();
      }
      int wc = shape_->width() * shape_->channel();
      int wc_aligned = align_image(wc);
      int remainder = wc_aligned - wc;

      char* src_start = data<char>();
      char* dst_start = dst_data;
      for (int n = 0; n < shape_->num(); n++) {
        for (int h = 0; h < shape_->height(); h++) {
          memcpy(dst_start, src_start, wc * cell_size);
          memset(dst_start + wc * cell_size, 0, remainder * cell_size);
          src_start += wc * cell_size;
          dst_start += wc_aligned * cell_size;
        }
      }
      if (dst == nullptr) {
        memcpy(data<void>(), dst_data, mem_size);
        flush();
        fpga_free(dst_data);
      } else {
        dst->flush();
      }
    } else {
      if (copy) {
        dst->copyFrom(this);
      } else {
        // TODO(chonwhite) share data.
      }
    }
  }

  void unalignImage(Tensor* dst = nullptr, bool copy = false) {
    if (shape_->shouldAlign()) {
      // int cell_size = CellSize(this->dataType_);
      // char* dst_data = nullptr;
      // size_t mem_size = shape_->memorySize(cell_size);
      // if (dst == nullptr) {
      //     dst_data = (char*)fpga_malloc(mem_size);
      // } else {
      //     dst_data = dst->data<char>();
      // }
      // int wc = shape_->width() * shape_->channel();
      // int wc_aligned = align_image(wc);
      // int remainder = wc_aligned - wc;

      // char* src_start = data<char>();
      // char* dst_start = dst_data;
      // for (int n = 0; n < shape_->num(); n++) {
      //     for (int h = 0;h < shape_->height(); h++) {
      //         memcpy(dst_start, src_start, wc * cell_size);
      //         memcpy(dst_start + wc * cell_size, 0, remainder * cell_size);
      //         src_start += wc * cell_size;
      //         dst_start += wc_aligned * cell_size;
      //     }
      // }
      // if (dst == nullptr) {
      //     memcpy(data<void>(), dst_data, mem_size);
      //     flush();
      //     fpga_free(dst_data);
      // } else {
      //     dst->flush();
      // }
    } else {
      if (copy) {
        dst->copyFrom(this);
      } else {
        // TODO(chonwhite) share data.
      }
    }
  }

  void copyFrom(Tensor* src) {
    BypassArgs args;
    args.input_data_type =
        src->dataType_ == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    args.output_data_type = dataType_ == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    args.input_layout_type = LAYOUT_HWC;
    args.output_layout_type = LAYOUT_HWC;
    args.image = {.address = src->data<void>(),
                  .scale_address = src->scale(),
                  .channels = (uint32_t)src->shape().numel(),
                  .width = 1,
                  .height = 1,
                  .pad_width = 0u,
                  .pad_height = 0u};
    args.output = {
        .address = data<void>(),
        .scale_address = scale(),
    };
    src->flush();
    perform_bypass(args);
    this->invalidate();
  }

  void flush() { fpga_flush(placeHolder_->data(), placeHolder_->memorySize()); }

  void invalidate() {
    fpga_invalidate(placeHolder_->data(), placeHolder_->memorySize());
  }

  void print() {
    int count = shape_->numel();
    for (int i = 0; i < count; i++) {
      std::cout << "" << '\n';
    }
  }

  std::string dimsFileName() {
    return std::to_string(shape_->num()) + "_" +
           std::to_string(shape_->channel()) + "_" +
           std::to_string(shape_->height()) + "_" +
           std::to_string(shape_->width()) + ".txt";
  }

  void saveToFile() {
    // std::string path = std::to_string(id_) + ".txt";
    std::string path = dimsFileName();
    saveToFile(path);
  }

  void saveToFile(std::string path) {
    // return;
    invalidate();
    std::ofstream ofs;

    static int counter = 0;
    std::string npath = std::to_string(counter) + "_" + path;
    counter++;
    ofs.open(npath);
    for (size_t i = 0; i < shape_->numel(); i++) {
      float value = 0;
      if (dataType_ == FP32) {
        value = data<float>()[i];
      } else {
        value = half_to_float(data<float16>()[i]);
      }
      ofs << value << std::endl;
    }
    ofs.close();
  }

  ~Tensor() {
    if (shape_ != nullptr) {
      delete shape_;
      shape_ = nullptr;
    }
    if (placeHolder_ != nullptr) {
      delete placeHolder_;
      placeHolder_ = nullptr;
    }
  }

 private:
  float scale_[2];
  Shape* shape_ = nullptr;
  DataType dataType_ = FP32;
  bool aligned_ = false;

  static int generateID() {
    static int sID = 0;
    int id = sID++;
    return id;
  }

  int id_ = generateID();

  PlaceHolder* placeHolder_ = nullptr;
};

}  // namespace zynqmp
}  // namespace paddle_mobile
