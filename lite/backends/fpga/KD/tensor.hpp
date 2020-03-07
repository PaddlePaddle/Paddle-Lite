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
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "lite/backends/fpga/KD/dl_engine.hpp"
#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/shape.hpp"

namespace paddle {
namespace zynqmp {

enum DataType : int {
  FP32 = 0,
  FP16 = 1,
  INT8 = 2,
  INT32 = 3,
};

enum DataSyncStatus : int {
  Synched = 0,
  Device = 1,
  CPU = 2,
};

typedef uint16_t float16;

inline int CellSize(DataType type) {
  switch (type) {
    case FP32:
      return sizeof(float);
    case FP16:
      return sizeof(float16);
    case INT32:
      return sizeof(int32_t);
    case INT8:
      return sizeof(int8_t);
    default:
      return 0;
  }
  return 0;
}

class PlaceHolder {
 public:
  PlaceHolder() {}
  explicit PlaceHolder(size_t size) {
    size_ = size;
    data_ = fpga_malloc(size_);
  }

  void* data() { return data_; }
  void set_data(const void* ptr) { data_ = const_cast<void*>(ptr); }

  size_t memorySize() { return size_; }
  void set_size(size_t new_size) { size_ = new_size; }

  ~PlaceHolder() { fpga_free(data_); }

  float scale_[2];

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

class Tensor {
 public:
  Tensor() { DLEngine::get_instance(); }

  int id() { return id_; }

  template <typename Dtype>
  Dtype* data() {
    if (placeHolder_ == nullptr) {
      return nullptr;
    }
    void* ptr = reinterpret_cast<char*>(this->placeHolder_->data()) +
                offset * CellSize(dataType_);
    return reinterpret_cast<Dtype*>(ptr);
  }

  template <typename Dtype>
  Dtype* mutableData(DataType dataType, const Shape& shape) {
    if (this->shape_ != nullptr) {
      delete shape_;
    }
    this->shape_ = new Shape(shape);
    this->dataType_ = dataType;
    return mutableData<Dtype>();
  }

  template <typename Dtype>
  Dtype* mutableData() {
    size_t memorySize =
        shape_->memorySize(CellSize(dataType_)) * mem_scale_factor_;
    if (placeHolder_ != nullptr) {
      if (memorySize > placeHolder_->memorySize()) {
        placeHolder_.reset(new PlaceHolder(memorySize));
      }
    } else {
      placeHolder_.reset(new PlaceHolder(memorySize));
    }
    return data<Dtype>();
  }

  size_t memorySize() {
    if (placeHolder_ == nullptr) {
      return 0;
    }
    return placeHolder_->memorySize();
  }

  void setDataType(DataType dataType) { this->dataType_ = dataType; }

  DataType dataType() { return this->dataType_; }

  Shape& shape() { return *shape_; }

  bool aligned() { return this->aligned_; }

  void setAligned(bool aligned) { this->aligned_ = aligned; }

  float* scale() { return placeHolder_->scale_; }

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
    if (dst != nullptr) {
      dst->copyScaleFrom(this);
    }
  }

  inline void copyScaleFrom(Tensor* src) {
    placeHolder_->scale_[0] = src->placeHolder_->scale_[0];
    placeHolder_->scale_[1] = src->placeHolder_->scale_[1];
  }

  void unalignImage(Tensor* dst = nullptr, bool copy = false) {
    Tensor* target = dst == nullptr ? this : dst;
    if (!target->aligned_) {
      if (copy && dst != nullptr) {
        dst->copyFrom(this);
      }
      return;
    }
    target->syncToCPU();
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

      char* src_start = data<char>();
      char* dst_start = dst_data;
      for (int n = 0; n < shape_->num(); n++) {
        for (int h = 0; h < shape_->height(); h++) {
          memcpy(dst_start, src_start, wc * cell_size);
          src_start += wc_aligned * cell_size;
          dst_start += wc * cell_size;
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

  void setMemScale(float scale_factor) {
    this->mem_scale_factor_ = scale_factor;
  }

  void shareDataWith(Tensor* src) { shareDataWith(src, src->shape()); }

  void shareDataWith(Tensor* src, const Shape& shape, int offset = 0) {
    if (shape_ != nullptr) {
      delete shape_;
    }
    this->placeHolder_ = src->placeHolder_;
    this->dataType_ = src->dataType_;
    this->aligned_ = src->aligned_;
    this->dateLocation_ = src->dateLocation_;
    this->offset = offset;
    shape_ = new Shape(const_cast<Shape&>(shape));
  }

  void copyFrom(Tensor* src) {
    if (src->dataType_ == dataType_) {
      src->syncToCPU();
      memcpy(data<void>(), src->data<void>(), memorySize());
      copyScaleFrom(src);
      flush();
      return;
    }
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

    ImageOutputArgs output = {
        .address = data<void>(), .scale_address = scale(),
    };
    args.output = output;
    src->syncToDevice();
    size_t aligned_remainder = src->shape().numel() % 16;
    if (aligned_remainder > 0) {
      size_t dtype_size =
          src->dataType_ == FP32 ? sizeof(float) : sizeof(float16);
      void* dst = src->data<char>() + src->shape().numel() * dtype_size;
      memset(dst, 0, aligned_remainder * dtype_size);
      fpga_flush(dst, aligned_remainder * dtype_size);
    }
    src->syncToDevice();
    this->invalidate();
    perform_bypass(args);
    this->invalidate();
  }

  void flush() {
    size_t memorySize = placeHolder_->memorySize();
    fpga_flush(placeHolder_->data(), memorySize);
  }

  void invalidate() {
    size_t memorySize = placeHolder_->memorySize();
    fpga_invalidate(placeHolder_->data(), memorySize);
  }

  void sync() {
    switch (synchedStatus_) {
      case CPU:
        flush();
        break;
      case Device:
        invalidate();
        break;
      default:
        break;
    }
  }

  void syncToCPU() {
    if (dateLocation_ == Device) {
      invalidate();
    }
  }

  void syncToDevice() {
    if (dateLocation_ == CPU) {
      flush();
    }
  }

  DataSyncStatus synchedStatus() { return synchedStatus_; }

  void setSynchedStatus(DataSyncStatus status) { synchedStatus_ = status; }

  void setDataLocation(DataSyncStatus location) { dateLocation_ = location; }

  void print() {}

  void printScale() {
    if (placeHolder_ == nullptr) {
      return;
    }
  }

  void printScale(std::string type) { printScale(); }

  std::string dimsFileName() {
    return std::to_string(shape_->num()) + "_" +
           std::to_string(shape_->channel()) + "_" +
           std::to_string(shape_->height()) + "_" +
           std::to_string(shape_->width()) + ".txt";
  }

  void saveToFile() { std::string path = dimsFileName(); }

  void saveToFile(std::string prefix, bool with_shape) {
    std::string path = prefix;
    if (with_shape) {
      path = path + "_" + dimsFileName();
    } else {
      path = path + ".txt";
    }
    saveToFile(path);
  }

  void saveToFile(std::string path) {
    syncToCPU();
    invalidate();
    std::ofstream ofs;
    static int counter = 0;
    std::string npath = std::to_string(counter) + "_" + path;
    counter++;
    save_file_with_name(npath);
  }

  void save_file_with_name(std::string path) {
    invalidate();
    std::ofstream ofs;
    ofs.open(path);
    ofs << scale()[0] << " / " << scale()[1] << std::endl;

    for (int i = 0; i < shape_->numel(); i++) {
      float value = 0;
      if (dataType_ == FP32) {
        value = data<float>()[i];
      } else if (dataType_ == FP16) {
        value = half_to_float(data<float16>()[i]);
      } else {
        value = data<int8_t>()[i];
      }
      ofs << value << std::endl;
    }
    ofs.close();
  }

  void readFromFile(std::string path) {
    std::ifstream file_stream;
    file_stream.open(path);
    if (!file_stream) {
      return;
    }
    int num = shape_->numel();
    invalidate();
    float max = 0.0f;
    if (dataType_ == FP16) {
      float16* data = mutableData<float16>();
      for (int i = 0; i < num; ++i) {
        float value = 0;
        file_stream >> value;
        max = std::max(std::abs(value), max);
        data[i] = float_to_half(value);
      }
    } else {
      float* data = mutableData<float>();
      for (int i = 0; i < num; ++i) {
        float value = 0;
        file_stream >> value;
        max = std::max(std::abs(value), max);
        data[i] = value;
      }
    }
    flush();
    placeHolder_->scale_[0] = max / 127.0f;
    placeHolder_->scale_[1] = 127.0f / max;
  }

  friend std::ostream& operator<<(std::ostream& os, Tensor& tensor) {
    os << "tensor:"
       << "\n";
    os << "dims: {";
    for (int i = 0; i < tensor.shape().dimSize(); ++i) {
      os << tensor.shape()[i] << " ";
    }
    os << "}\n";
    for (int i = 0; i < tensor.shape().numel(); i++) {
      float value = 0;
      if (tensor.dataType() == FP32) {
        value = tensor.data<float>()[i];
      } else {
        value = half_to_float(tensor.data<float16>()[i]);
      }
      os << value << " ";
    }
    os << "\n";
    return os;
  }

  ~Tensor() {
    if (shape_ != nullptr) {
      delete shape_;
      shape_ = nullptr;
    }
  }

 private:
  int offset = 0;
  float mem_scale_factor_ = 1.0f;
  std::shared_ptr<PlaceHolder> placeHolder_;
  Shape* shape_ = nullptr;
  DataType dataType_ = FP32;
  bool aligned_ = false;
  DataSyncStatus synchedStatus_ = Synched;
  DataSyncStatus dateLocation_ = Device;

  static int generateID() {
    static int sID = 0;
    int id = sID++;
    return id;
  }

  int id_ = generateID();
};

}  // namespace zynqmp
}  // namespace paddle
