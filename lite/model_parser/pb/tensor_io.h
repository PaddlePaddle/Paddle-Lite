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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/model/base/io.h"
#include "lite/core/model/base/param_desc.h"
#include "lite/core/scope.h"
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/model_parser/pb/param_desc.h"
#endif  // LITE_ON_TINY_PUBLISH

namespace paddle {
namespace lite {
namespace model_parser {
namespace tensor {

void set_lod(lite::Tensor* tensor,
             const std::vector<std::vector<uint64_t>>& lod);

const std::vector<std::vector<uint64_t>>& get_lod(const lite::Tensor& tensor);

void set_allocation(lite::Tensor* tensor,
                    const std::vector<int64_t>& shape,
                    paddle::lite_api::PrecisionType precision);

void set_allocation(const lite::Tensor& tensor,
                    TensorInfoWriteAPI* tensor_info);

size_t get_bytes_size(const lite::Tensor& tensor);

void* get_allocation(lite::Tensor* tensor);

const void* get_allocation(const lite::Tensor& tensor);
}  // namespace tensor

namespace pb {
class LoDTensorDeserializer {
 public:
  LoDTensorDeserializer() : buf_(new Buffer) {}

  void ForwardRead(lite::Tensor* tensor, ByteReader* reader);

 private:
  std::unique_ptr<Buffer> buf_;
};

#ifndef LITE_ON_TINY_PUBLISH
class LoDTensorSerializer {
 public:
  LoDTensorSerializer() : buf_(new Buffer) {}
  void ForwardWrite(const lite::Tensor& tensor,
                    ByteWriter* writer,
                    uint32_t version = 0);

 private:
  std::unique_ptr<Buffer> buf_;
};
#endif  // LITE_ON_TINY_PUBLISH

}  // namespace pb
}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
