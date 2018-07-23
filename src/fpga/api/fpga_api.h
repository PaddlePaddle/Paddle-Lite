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

#pragma once

#include <cstddef>
#include <iostream>
#include <limits>

// memory management;

namespace paddle {
namespace mobile {
namespace fpga {
namespace api {

int open_device();
int close_device();

void *fpga_malloc(size_t size);
void fpga_free(void *ptr);
void fpga_copy(void *dst, const void *src, size_t num);

struct CnnVersionArgs {
  void *buf;
};

struct QuantArgs {
  float scale;
};

struct BatchNormalizationArgs {
  bool enable;
};

struct ScaleArgs {};

#define IOCTL_CNN_MAGIC 'CNN'
#define IOCTL_VERSION _IOW(IOCTL_CNN_MAGIC, 1, struct CnnVersionArgs)
#define IOCTL_GET_QUANT _IOW(IOCTL_CNN_MAGIC, 2, struct QuantArgs)
#define IOCTL_SET_QUANT _IOW(IOCTL_CNN_MAGIC, 3, struct QuantArgs)

}  // namespace api
}  // namespace fpga
}  // namespace mobile
}  // namespace paddle
